import inspect
import json

import keras.layers as kl
import keras.optimizers as ko
import keras.regularizers as kreg
from keras import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from neural_LM.vocabulary import Vocabulary, FeatureVocabulary, vocabulary_from_json
from neural_LM.neural_lm import make_bucket_indexes
from neural_LM.common import *
from neural_tagging.cells import Highway

BUCKET_SIZE = 32
MAX_WORD_LENGTH = 30


def load_tagger(infile):
    with open(infile, "r", encoding="utf8") as fin:
        json_data = json.load(fin)
    args = {key: value for key, value in json_data.items()
            if not (key.endswith("_") or key.endswith("callback") or
                    key == "dump_file")}
    callbacks = []
    early_stopping_callback_data = json_data.get("early_stopping_callback")
    if early_stopping_callback_data is not None:
        callbacks.append(EarlyStopping(**early_stopping_callback_data))
    lr_callback_data = json_data.get("LR_callback")
    if lr_callback_data is not None:
        callbacks.append(ReduceLROnPlateau(**lr_callback_data))
    model_checkpoint_callback_data = json_data.get("model_checkpoint_callback")
    if model_checkpoint_callback_data is not None:
        model_checkpoint_callback_data["save_weights_only"] = True
        model_checkpoint_callback_data["save_best_only"] = True
        callbacks.append(ModelCheckpoint(**model_checkpoint_callback_data))
    args['callbacks'] = callbacks
    # создаём языковую модель
    tagger = CharacterTagger(**args)
    # обучаемые параметры
    args = {key: value for key, value in json_data.items() if key[-1] == "_"}
    for key, value in args.items():
        if key == "symbols_":
            value = vocabulary_from_json(value)
        elif key == "tags_":
            value = vocabulary_from_json(value, use_features=True)
        setattr(tagger, key, value)
    # модель
    tagger.build()  # не работает сохранение модели, приходится сохранять только веса
    tagger.model_.load_weights(json_data['dump_file'])
    return tagger


class CharacterTagger:
    """
    A class for character-based neural morphological tagger
    """
    def __init__(self, reverse=False, word_rnn="cnn",  min_char_count=1,
                 char_embeddings_size=16, char_conv_layers=1,
                 char_window_size=5, char_filters=None,
                 char_filter_multiple=25, char_highway_layers=1,
                 conv_dropout=0.0, highway_dropout=0.0,
                 intermediate_dropout=0.0, lstm_dropout=0.0,
                 word_lstm_layers=1, word_lstm_units=128,
                 word_dropout=0.0, regularizer=None,
                 batch_size=16, validation_split=0.2, nepochs=25,
                 min_prob=0.01, max_diff=2.0,
                 callbacks=None, verbose=1):
        self.reverse = reverse
        self.word_rnn = word_rnn
        self.min_char_count = min_char_count
        self.char_embeddings_size = char_embeddings_size
        self.char_conv_layers = char_conv_layers
        self.char_window_size = char_window_size
        self.char_filters = char_filters
        self.char_filter_multiple = char_filter_multiple
        self.char_highway_layers = char_highway_layers
        self.conv_dropout = conv_dropout
        self.highway_dropout = highway_dropout
        self.intermediate_dropout = intermediate_dropout
        self.word_lstm_layers = word_lstm_layers
        self.word_lstm_units = word_lstm_units
        self.lstm_dropout = lstm_dropout
        self.word_dropout = word_dropout
        self.regularizer = regularizer
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.nepochs=nepochs
        self.min_prob = min_prob
        self.max_diff = max_diff
        self.callbacks = callbacks
        self.verbose = verbose
        self.initialize()

    def initialize(self):
        if isinstance(self.char_window_size, int):
            self.char_window_size = [self.char_window_size]
        if self.char_filters is None or isinstance(self.char_filters, int):
            self.char_filters = [self.char_filters] * len(self.char_window_size)
        if len(self.char_window_size) != len(self.char_filters):
            raise ValueError("There should be the same number of window sizes and filter sizes")
        if isinstance(self.word_lstm_units, int):
            self.word_lstm_units = [self.word_lstm_units] * self.word_lstm_layers
        if len(self.word_lstm_units) != self.word_lstm_layers:
            raise ValueError("There should be the same number of lstm layer units and lstm layers")
        if self.regularizer is not None:
            self.regularizer = kreg.l2(self.regularizer)
        if self.fusion_regularizer is not None:
            self.fusion_regularizer = kreg.l2(self.fusion_regularizer)

    def to_json(self, outfile, model_file, lm_file=None):
        info = dict()
        if lm_file is not None:
            info["lm_file"] = lm_file
        # model_file = os.path.abspath(model_file)
        for (attr, val) in inspect.getmembers(self):
            if not (attr.startswith("__") or inspect.ismethod(val) or
                    isinstance(getattr(CharacterTagger, attr, None), property) or
                    isinstance(val, np.ndarray) or isinstance(val, Vocabulary) or
                    attr.isupper() or
                    attr in ["callbacks", "model_",  "regularizer"]):
                info[attr] = val
            elif isinstance(val, Vocabulary):
                info[attr] = val.jsonize()
            elif isinstance(val, np.ndarray):
                val = val.tolist()
                info[attr] = val
            elif attr == "model_":
                info["dump_file"] = model_file
                self.model_.save_weights(model_file)
            elif attr == "callbacks":
                for callback in val:
                    if isinstance(callback, EarlyStopping):
                        info["early_stopping_callback"] = {"patience": callback.patience,
                                                           "monitor": callback.monitor}
                    elif isinstance(callback, ModelCheckpoint):
                        info["model_checkpoint_callback"] =\
                            {key: getattr(callback, key) for key in ["monitor", "filepath"]}
                    elif isinstance(callback, ReduceLROnPlateau):
                        info["LR_callback"] =\
                            {key: getattr(callback, key) for key in
                             ["monitor", "factor", "patience", "cooldown", "epsilon"]}
            elif attr.endswith("regularizer"):
                if val is not None:
                    info[attr] = float(val.l2)
        with open(outfile, "w", encoding="utf8") as fout:
            json.dump(info, fout)

    @property
    def symbols_number_(self):
        return self.symbols_.symbols_number_

    @property
    def tags_number_(self):
        return self.tags_.symbols_number_

    def transform(self, data, labels=None, pad=True, return_indexes=True,
                  buckets_number=None, bucket_size=None, join_buckets=True):
        lengths = [len(x)+2 for x in data]
        if pad:
            indexes, level_lengths = make_bucket_indexes(
                lengths, buckets_number=buckets_number,
                bucket_size=bucket_size, join_buckets=join_buckets)
        else:
            indexes = [[i] for i in range(len(data))]
            level_lengths = lengths
        X = [None] * len(data)
        for bucket_indexes, bucket_length in zip(indexes, level_lengths):
            for i in bucket_indexes:
                sent = data[i] if not self.reverse else data[i][::-1]
                X[i] = [self._make_sent_vector(sent, bucket_length=bucket_length)]
                if labels is not None:
                    tags = labels[i] if not self.reverse else labels[i][::-1]
                    X[i].append(self._make_tags_vector(tags, bucket_length=bucket_length))
        if return_indexes:
            return X, indexes
        else:
            return X

    def _make_sent_vector(self, sent, bucket_length=None):
        if bucket_length is None:
            bucket_length = len(sent)
        answer = np.zeros(shape=(bucket_length, MAX_WORD_LENGTH+2), dtype=np.int32)
        for i, word in enumerate(sent):
            answer[i, 0] = BEGIN
            m = min(len(word), MAX_WORD_LENGTH)
            for j, x in enumerate(word[-m:]):
                answer[i, j+1] = self.symbols_.toidx(x)
            answer[i, m+1] = END
            answer[i, m+2:] = PAD
        return answer

    def _make_tags_vector(self, tags, bucket_length=None, func=None):
        m = len(tags)
        if bucket_length is None:
            bucket_length = m
        answer = np.zeros(shape=(bucket_length,), dtype=np.int32)
        for i, tag in enumerate(tags):
            answer[i] = self.tags_.toidx(tag) if func is None else func(tag)
        return answer

    def train(self, data, labels, dev_data=None, dev_labels=None,
              symbol_vocabulary_file=None, tags_vocabulary_file=None,
              lm_file=None, model_file=None, save_file=None):
        """
        Trains the tagger on data :data: with labels :labels:

        data: list of lists of sequences, a list of sentences
        labels: list of lists of strs,
            a list of sequences of tags, each tag is a feature-value structure
        :return:
        """
        if symbol_vocabulary_file is None:
            self.symbols_ = Vocabulary(
                character=True, min_count=self.min_char_count).train(data)
        else:
            self.symbols_ = vocabulary_from_json(
                symbol_vocabulary_file, use_features=False)
        if tags_vocabulary_file is None:
            self.tags_ = FeatureVocabulary(character=False).train(labels)
        else:
            with open(tags_vocabulary_file, "r", encoding="utf8") as fin:
                tags_info = json.load(fin)
            self.tags_ = vocabulary_from_json(tags_info, use_features=True)
        if self.verbose > 0:
            print("{} characters, {} tags".format(self.symbols_number_, self.tags_number_))
        X_train, indexes_by_buckets = self.transform(data, labels, buckets_number=10)
        if dev_data is not None:
            X_dev, dev_indexes_by_buckets =\
                self.transform(dev_data, dev_labels, bucket_size=BUCKET_SIZE)
        else:
            X_dev, dev_indexes_by_buckets = [None] * 2
        self.build()
        if save_file is not None and model_file is not None:
            self.to_json(save_file, model_file, lm_file)
        self._train_on_data(X_train, indexes_by_buckets, X_dev,
                            dev_indexes_by_buckets, model_file=model_file)
        return self

    def _train_on_data(self, X, indexes_by_buckets, X_dev=None,
                       dev_indexes_by_buckets=None, model_file=None):
        if X_dev is None:
            X_dev, dev_indexes_by_buckets = X, []
            validation_split = self.validation_split
        else:
            validation_split = 0.0
        train_indexes_by_buckets = []
        for curr_indexes in indexes_by_buckets:
            np.random.shuffle(curr_indexes)
            if validation_split != 0.0:
                train_bucket_size = int((1.0 - self.validation_split) * len(curr_indexes))
                train_indexes_by_buckets.append(curr_indexes[:train_bucket_size])
                dev_indexes_by_buckets.append(curr_indexes[train_bucket_size:])
            else:
                train_indexes_by_buckets.append(curr_indexes)
        if model_file is not None:
            callback = ModelCheckpoint(model_file, monitor="val_acc",
                                       save_weights_only=True, save_best_only=True)
            if self.callbacks is not None:
                self.callbacks.append(callback)
            else:
                self.callbacks = [callback]
        train_steps = sum((1 + (len(x)-1) // self.batch_size) for x in train_indexes_by_buckets)
        dev_steps = len(dev_indexes_by_buckets)
        train_gen = generate_data(X, train_indexes_by_buckets, self.tags_number_,
                                  self.batch_size, use_last=False)
        dev_gen = generate_data(X_dev, dev_indexes_by_buckets, self.tags_number_,
                                use_last=False, shuffle=False)
        self.model_.fit_generator(
            train_gen, steps_per_epoch=train_steps, epochs=self.nepochs,
            callbacks=self.callbacks, validation_data=dev_gen,
            validation_steps=dev_steps, verbose=1)
        if model_file is not None:
            self.model_.load_weights(model_file)
        return self

    def predict(self, data, labels=None, return_probs=False):
        X_test, indexes_by_buckets =\
            self.transform(data, labels=labels, bucket_size=BUCKET_SIZE)
        answer, probs = [None] * len(data), [None] * len(data)
        for k, (X_curr, bucket_indexes) in enumerate(
                zip(X_test[::-1], indexes_by_buckets[::-1])):
            X_curr = [np.array([X_test[i][j] for i in bucket_indexes])
                      for j in range(len(X_test[0])-int(labels is not None))]
            bucket_probs = self.model_.predict(X_curr, batch_size=256)
            bucket_labels = np.argmax(bucket_probs, axis=-1)
            for curr_labels, curr_probs, index in\
                    zip(bucket_labels, bucket_probs, bucket_indexes):
                curr_labels = curr_labels[:len(data[index])]
                curr_labels = [self.tags_.symbols_[label] for label in curr_labels]
                answer[index], probs[index] = curr_labels, curr_probs[:len(data[index])]
        return (answer, probs) if return_probs else answer

    def score(self, data, labels):
        X_test, indexes_by_buckets = self.transform(data, labels, bucket_size=BUCKET_SIZE)
        probs = [None] * len(data)
        for k, (X_curr, bucket_indexes) in enumerate(zip(X_test[::-1], indexes_by_buckets[::-1])):
            X_curr = [np.array([X_test[i][j] for i in bucket_indexes])
                      for j in range(len(X_test[0])-1)]
            y_curr = [np.array(X_test[i][-1]) for i in bucket_indexes]
            bucket_probs = self.model_.predict(X_curr, batch_size=256)
            for curr_labels, curr_probs, index in zip(y_curr, bucket_probs, bucket_indexes):
                L = len(data[index])
                probs[index] = curr_probs[np.arange(L), curr_labels[:L]]
        return probs

    def build(self):
        word_inputs = kl.Input(shape=(None, MAX_WORD_LENGTH+2), dtype="int32")
        inputs = [word_inputs]
        word_outputs = self.build_word_cnn(word_inputs)
        outputs, lstm_outputs = self.build_basic_network(word_outputs)
        compile_args = {"optimizer": ko.nadam(lr=0.002, clipnorm=5.0),
                        "loss": "categorical_crossentropy", "metrics": ["accuracy"]}
        self.model_ = Model(inputs, outputs)
        self.model_.compile(**compile_args)
        if self.verbose > 0:
            print(self.model_.summary())
        return self

    def build_word_cnn(self, inputs):
        # inputs = kl.Input(shape=(MAX_WORD_LENGTH,), dtype="int32")
        inputs = kl.Lambda(kb.one_hot, arguments={"num_classes": self.symbols_number_},
                           output_shape=lambda x: tuple(x) + (self.symbols_number_,))(inputs)
        char_embeddings = kl.Dense(self.char_embeddings_size, use_bias=False)(inputs)
        conv_outputs = []
        self.char_output_dim_ = 0
        for window_size, filters_number in zip(self.char_window_size, self.char_filters):
            curr_output = char_embeddings
            curr_filters_number = (min(self.char_filter_multiple * window_size, 200)
                                   if filters_number is None else filters_number)
            for _ in range(self.char_conv_layers - 1):
                curr_output = kl.Conv2D(curr_filters_number, (1, window_size),
                                        padding="same", activation="relu",
                                        data_format="channels_last")(curr_output)
                if self.conv_dropout > 0.0:
                    curr_output = kl.Dropout(self.conv_dropout)(curr_output)
            curr_output = kl.Conv2D(curr_filters_number, (1, window_size),
                                    padding="same", activation="relu",
                                    data_format="channels_last")(curr_output)
            conv_outputs.append(curr_output)
            self.char_output_dim_ += curr_filters_number
        if len(conv_outputs) > 1:
            conv_output = kl.Concatenate(axis=-1)(conv_outputs)
        else:
            conv_output = conv_outputs[0]
        highway_input = kl.Lambda(kb.max, arguments={"axis": -2})(conv_output)
        if self.intermediate_dropout > 0.0:
            highway_input = kl.Dropout(self.intermediate_dropout)(highway_input)
        for i in range(self.char_highway_layers - 1):
            highway_input = Highway(activation="relu")(highway_input)
            if self.highway_dropout > 0.0:
                highway_input = kl.Dropout(self.highway_dropout)(highway_input)
        highway_output = Highway(activation="relu")(highway_input)
        return highway_output

    def build_basic_network(self, word_outputs):
        """
        Creates the basic network architecture,
        transforming word embeddings to intermediate outputs
        """
        if self.word_dropout > 0.0:
            lstm_outputs = kl.Dropout(self.word_dropout)(word_outputs)
        else:
            lstm_outputs = word_outputs
        for j in range(self.word_lstm_layers-1):
            lstm_outputs = kl.Bidirectional(
                kl.LSTM(self.word_lstm_units[j], return_sequences=True,
                        dropout=self.lstm_dropout))(lstm_outputs)
        lstm_outputs = kl.Bidirectional(
                kl.LSTM(self.word_lstm_units[-1], return_sequences=True,
                        dropout=self.lstm_dropout))(lstm_outputs)
        pre_outputs = kl.TimeDistributed(
                kl.Dense(self.tags_number_, activation="softmax",
                         activity_regularizer=self.regularizer),
                name="p")(lstm_outputs)
        return pre_outputs, lstm_outputs





