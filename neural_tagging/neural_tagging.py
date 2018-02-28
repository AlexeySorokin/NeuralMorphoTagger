from collections import defaultdict
import inspect
import json
import os

import numpy as np

import keras.backend as kb
import keras.layers as kl
import keras.optimizers as ko
import keras.regularizers as kreg
from keras import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from neural_LM.vocabulary import Vocabulary, FeatureVocabulary, vocabulary_from_json
from neural_LM.neural_lm import make_bucket_indexes
from neural_LM.common import *
from neural_LM import NeuralLM, load_lm
from neural_tagging.cells import Highway, WeightedCombinationLayer,\
    TemporalDropout, leader_loss, positions_func

BUCKET_SIZE = 32
MAX_WORD_LENGTH = 30


def load_tagger(infile):
    with open(infile, "r", encoding="utf8") as fin:
        json_data = json.load(fin)
    args = {key: value for key, value in json_data.items()
            if not (key.endswith("_") or key.endswith("callback") or
                    key in ["dump_file", "lm_file"])}
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
        elif key == "tag_embeddings_":
            value = np.asarray(value)
        setattr(tagger, key, value)
    # loading language model
    if tagger.use_lm and "lm_file" in json_data:
        tagger.lm_ = load_lm(json_data["lm_file"])
    # модель
    tagger.build()  # не работает сохранение модели, приходится сохранять только веса
    tagger.model_.load_weights(json_data['dump_file'])
    return tagger

class CharacterTagger:
    """
    A class for character-based neural morphological tagger
    """
    def __init__(self, reverse=False, use_lm_loss=False, use_lm=False,
                 normalize_lm_embeddings=False, base_model_weight=0.25,
                 word_rnn="cnn", min_char_count=1, char_embeddings_size=16,
                 char_conv_layers=1, char_window_size=5,
                 char_filters=None, char_filter_multiple=25,
                 char_highway_layers=1, conv_dropout=0.0, highway_dropout=0.0,
                 intermediate_dropout=0.0, word_dropout=0.0, lm_dropout=0.0,
                 word_lstm_layers=1, word_lstm_units=128, lstm_dropout=0.0,
                 use_rnn_for_weight_state=False, weight_state_rnn_units=64,
                 use_fusion=False, fusion_state_units=256, use_dimension_bias=False,
                 use_intermediate_activation_for_weights=False,
                 intermediate_units_for_weights=64,
                 use_leader_loss=False, leader_loss_weight=0.2,
                 regularizer=None, fusion_regularizer=None,
                 probs_threshold=None, lm_probs_threshold=None,
                 batch_size=16, validation_split=0.2, nepochs=25,
                 min_prob=0.01, max_diff=2.0,
                 callbacks=None, verbose=1):
        self.reverse = reverse
        self.use_lm_loss = use_lm_loss
        self.use_lm = use_lm
        self.normalize_lm_embeddings = normalize_lm_embeddings
        self.base_model_weight = base_model_weight
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
        self.word_dropout = word_dropout
        self.word_lstm_layers = word_lstm_layers
        self.word_lstm_units = word_lstm_units
        self.lstm_dropout = lstm_dropout
        self.lm_dropout = lm_dropout
        self.use_rnn_for_weight_state = use_rnn_for_weight_state
        self.weight_state_rnn_units = weight_state_rnn_units
        self.use_fusion = use_fusion
        self.fusion_state_units = fusion_state_units
        self.use_dimension_bias = use_dimension_bias
        self.use_intermediate_activation_for_weights = use_intermediate_activation_for_weights
        self.intermediate_units_for_weights = intermediate_units_for_weights
        self.use_leader_loss = use_leader_loss
        self.leader_loss_weight = leader_loss_weight
        self.regularizer = regularizer
        self.fusion_regularizer = fusion_regularizer
        self.probs_threshold = probs_threshold
        self.lm_probs_threshold = lm_probs_threshold
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
                    attr in ["callbacks", "model_", "_basic_model_",
                             "_decoder_", "lm_", "regularizer", "fusion_regularizer"]):
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
                length_func = lambda x: min(len(x), MAX_WORD_LENGTH)+2
                X[i] = [self._make_sent_vector(sent, bucket_length=bucket_length)]
                # X[i] = [self._make_sent_vector(sent, bucket_length=bucket_length),
                #         self._make_tags_vector(sent, bucket_length=bucket_length,
                #                                func=length_func)]
                if labels is not None:
                    tags = labels[i] if not self.reverse else labels[i][::-1]
                    X[i].append(self._make_tags_vector(tags, bucket_length=bucket_length))
            if labels is not None and hasattr(self, "lm_"):
                curr_bucket = np.array([X[i][-1] for i in bucket_indexes])
                padding = np.full((curr_bucket.shape[0], 1), BEGIN, dtype=int)
                curr_bucket = np.hstack((padding, curr_bucket[:,:-1]))
                # transforming indexes to features
                curr_bucket = self.lm_.vocabulary_.symbol_matrix_[curr_bucket]
                lm_probs = self.lm_.model_.predict(curr_bucket)
                lm_states = self.lm_.hidden_state_func_([curr_bucket, 0])[0]
                self.lm_state_dim_ = lm_states.shape[-1]
                for i, index in enumerate(bucket_indexes):
                    X[index].insert(1, lm_states[i])
                    if not self.use_fusion:
                        X[index].insert(1, lm_probs[i])
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
            self.symbols_ = Vocabulary(character=True, min_count=self.min_char_count).train(data)
        else:
            self.symbols_ = vocabulary_from_json(symbol_vocabulary_file, use_features=False)
        if tags_vocabulary_file is None:
            self.tags_ = FeatureVocabulary(character=False).train(labels)
        else:
            with open(tags_vocabulary_file, "r", encoding="utf8") as fin:
                tags_info = json.load(fin)
            self.tags_ = vocabulary_from_json(tags_info, use_features=True)
        if lm_file is not None and (self.use_lm or self.use_lm_loss):
            lm = load_lm(lm_file)
        if lm_file is not None and self.use_lm_loss:
            self._make_tag_embeddings(lm)
        if lm_file is not None and self.use_lm:
            self.lm_ = lm
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
            monitor = "val_p_output_acc" if self.use_lm else "val_acc"
            callback = ModelCheckpoint(model_file, monitor=monitor,
                                       save_weights_only=True, save_best_only=True)
            if self.callbacks is not None:
                self.callbacks.append(callback)
            else:
                self.callbacks = [callback]
        train_steps = sum((1 + (len(x)-1) // self.batch_size) for x in train_indexes_by_buckets)
        dev_steps = len(dev_indexes_by_buckets)
        train_gen = generate_data(X, train_indexes_by_buckets, self.tags_number_,
                                  self.batch_size, use_last=False, duplicate_answer=self.use_lm)
        dev_gen = generate_data(X_dev, dev_indexes_by_buckets, self.tags_number_,
                                use_last=False, shuffle=False, duplicate_answer=self.use_lm)
        self.model_.fit_generator(
            train_gen, steps_per_epoch=train_steps, epochs=self.nepochs,
            callbacks=self.callbacks, validation_data=dev_gen,
            validation_steps=dev_steps, verbose=1)
        if model_file is not None:
            self.model_.load_weights(model_file)
        return self

    def predict(self, data, labels=None, beam_width=1,
                return_probs=False, return_basic_probs=False):
        X_test, indexes_by_buckets =\
            self.transform(data, labels=labels, bucket_size=BUCKET_SIZE)
        answer, probs = [None] * len(data), [None] * len(data)
        basic_probs = [None] * len(data)
        for k, (X_curr, bucket_indexes) in enumerate(zip(X_test[::-1], indexes_by_buckets[::-1])):
            X_curr = [np.array([X_test[i][j] for i in bucket_indexes])
                      for j in range(len(X_test[0])-int(labels is not None))]
            if self.use_lm and labels is None:
                print("Bucket {} of {} predicting".format(k+1, len(indexes_by_buckets)))
                batch_answer = self.predict_on_batch(X_curr[0], beam_width)
                bucket_labels = [x[0] for x in batch_answer[0]]
                bucket_probs = [x[0] for x in batch_answer[1]]
                bucket_basic_probs = [x[0] for x in batch_answer[2]]
            else:
                bucket_probs = self.model_.predict(X_curr, batch_size=256)
                bucket_basic_probs = [None] * len(bucket_indexes)
                if isinstance(bucket_probs, list):
                    bucket_probs, bucket_basic_probs = bucket_probs
                bucket_labels = np.argmax(bucket_probs, axis=-1)
            for curr_labels, curr_probs, curr_basic_probs, index in\
                    zip(bucket_labels, bucket_probs, bucket_basic_probs, bucket_indexes):
                curr_labels = curr_labels[:len(data[index])]
                curr_labels = [self.tags_.symbols_[label] for label in curr_labels]
                answer[index], probs[index] = curr_labels, curr_probs[:len(data[index])]
                basic_probs[index] = curr_basic_probs
        return ((answer, probs, basic_probs) if (return_basic_probs and self.use_lm)
                else (answer, probs) if return_probs else answer)

    def score(self, data, labels, return_basic_probs=False):
        X_test, indexes_by_buckets = self.transform(data, labels, bucket_size=BUCKET_SIZE)
        probs, basic_probs = [None] * len(data), [None] * len(data)
        for k, (X_curr, bucket_indexes) in enumerate(zip(X_test[::-1], indexes_by_buckets[::-1])):
            X_curr = [np.array([X_test[i][j] for i in bucket_indexes])
                      for j in range(len(X_test[0])-1)]
            y_curr = [np.array(X_test[i][-1]) for i in bucket_indexes]
            bucket_probs = self.model_.predict(X_curr, batch_size=256)
            if self.use_lm:
                bucket_probs, bucket_basic_probs = bucket_probs
            else:
                bucket_basic_probs = [None] * len(bucket_indexes)
            for curr_labels, curr_probs, curr_basic_probs, index in\
                    zip(y_curr, bucket_probs, bucket_basic_probs, bucket_indexes):
                L = len(data[index])
                probs[index] = curr_probs[np.arange(L), curr_labels[:L]]
                if curr_basic_probs is not None:
                    basic_probs[index] = curr_basic_probs[np.arange(L), curr_labels[:L]]
        return (probs, basic_probs) if (return_basic_probs and self.use_lm) else probs

    def predict_on_batch(self, X, beam_width=1, return_log=False):
        m, L = X.shape[:2]
        basic_outputs = self._basic_model_([X, 0])
        M = m * beam_width
        for i, elem in enumerate(basic_outputs):
            basic_outputs[i] = np.repeat(elem, beam_width, axis=0)
        # positions[j] --- текущая позиция в symbols[j]
        tags, probs, basic_probs = [[] for _ in range(M)], [[] for _ in range(M)], [[] for _ in range(M)]
        partial_scores = np.zeros(shape=(M,), dtype=float)
        is_active, active_count = np.zeros(dtype=bool, shape=(M,)), m
        is_completed = np.zeros(dtype=bool, shape=(M,))
        is_active[np.arange(0, M, beam_width)] = True
        lm_inputs = np.full(shape=(M, L+1), fill_value=BEGIN, dtype=np.int32)
        lm_inputs = self.tags_.symbol_matrix_[lm_inputs]
        # определяем функцию удлинения входа для языковой модели
        def lm_inputs_func(old, new, i, matrix):
            return np.concatenate([old[:i+1], [matrix[new]], old[i+2:]])
        for i in range(L):
            for j, start in enumerate(range(0, M, beam_width)):
                if np.max(X[j, i]) == 0 and is_active[start]:
                    # complete the sequences not completed yet
                    # is_active[start] checks that the group has not yet been completed
                    is_completed[start:start+beam_width] = is_active[start:start+beam_width]
                    is_active[start:start+beam_width] = False
            if not any(is_active):
                break
            active_lm_inputs = lm_inputs[is_active,:i+1]
            # predicting language model probabilities and states
            # if not self.use_fusion:
            #     lm_probs, lm_states = self.lm_.model_.predict([active_lm_inputs])
            # lm_states = self.lm_.hidden_state_func_([active_lm_inputs, 0])[0]
            if not self.use_fusion:
                lm_probs, lm_states = self.lm_.output_and_state_func_([active_lm_inputs, 0])
            else:
                lm_states = self.lm_.hidden_state_func_([active_lm_inputs, 0])[0]
            # keeping only active basic outputs
            active_basic_outputs = [elem[is_active,i:i+1] for elem in basic_outputs]
            if self.use_fusion:
                final_layer_inputs = active_basic_outputs + [lm_states[:,-1:]]
            else:
                final_layer_inputs = active_basic_outputs + [lm_probs[:,-1:], lm_states[:,-1:]]
            final_layer_outputs = self._decoder_(final_layer_inputs + [0])[0][:,0]
            hypotheses_by_groups = [[] for _ in range(m)]
            if beam_width == 1:
                curr_output_tags = np.argmax(final_layer_outputs, axis=-1)
                for r, (j, curr_probs, curr_basic_probs, index) in\
                        enumerate(zip(np.nonzero(is_active)[0], final_layer_outputs,
                                      active_basic_outputs[0], curr_output_tags)):
                    new_score = partial_scores[j] - np.log10(curr_probs[index])
                    hyp = (j, index, new_score, -np.log10(curr_probs[index]),
                           -np.log10(curr_basic_probs[0, index]))
                    hypotheses_by_groups[j] = [hyp]
            else:
                curr_best_scores = [np.inf] * m
                for r, (j, curr_probs, curr_basic_probs) in enumerate(
                        zip(np.nonzero(is_active)[0], final_layer_outputs, active_basic_outputs[0])):
                    group_index = j // beam_width
                    prev_partial_score = partial_scores[j]
                    # переходим к логарифмической шкале
                    curr_probs = -np.log10(np.clip(curr_probs, EPS, 1.0))
                    curr_basic_probs = -np.log10(np.clip(curr_basic_probs, EPS, 1.0))
                    if np.isinf(curr_best_scores[group_index]):
                        curr_best_scores[group_index] = prev_partial_score + np.min(curr_probs)
                    min_log_prob = curr_best_scores[group_index] - prev_partial_score + self.max_diff
                    min_log_prob = min(-np.log10(self.min_prob), min_log_prob)
                    possible_indexes = np.where(curr_probs <= min_log_prob)[0]
                    if len(possible_indexes) == 0:
                        possible_indexes = [np.argmin(curr_probs)]
                    for index in possible_indexes:
                        new_score = prev_partial_score + curr_probs[index]
                        hyp = (j, index, new_score, curr_probs[index], curr_basic_probs[0, index])
                        hypotheses_by_groups[group_index].append(hyp)
            for j, curr_hypotheses in enumerate(hypotheses_by_groups):
                curr_hypotheses = sorted(curr_hypotheses, key=(lambda x:x[2]))[:beam_width]
                group_start = j * beam_width
                is_active[group_start:group_start+beam_width] = False
                group_indexes = np.arange(group_start, group_start+len(curr_hypotheses))
                extend_history(tags, curr_hypotheses, group_indexes, pos=1)
                extend_history(probs, curr_hypotheses, group_indexes, pos=3)
                extend_history(basic_probs, curr_hypotheses, group_indexes, pos=4)
                extend_history(partial_scores, curr_hypotheses, group_indexes,
                               pos=2, func=lambda x, y: y)
                extend_history(lm_inputs, curr_hypotheses, group_indexes, pos=1,
                               func=lm_inputs_func, i=i, matrix=self.tags_.symbol_matrix_)
                is_active[group_indexes] = True
        # здесь нужно переделать words, probs в список
        tags_by_groups, probs_by_groups, basic_probs_by_groups = [], [], []
        for group_start in range(0, M, beam_width):
            # приводим к списку, чтобы иметь возможность сортировать
            active_indexes_for_group = list(np.where(is_completed[group_start:group_start+beam_width])[0])
            tags_by_groups.append([tags[group_start+i] for i in active_indexes_for_group])
            curr_group_probs = [np.array(probs[group_start+i])
                                for i in active_indexes_for_group]
            curr_basic_group_probs = [np.array(basic_probs[group_start+i])
                                      for i in active_indexes_for_group]
            if not return_log:
                curr_group_probs = [np.power(10.0, -elem) for elem in curr_group_probs]
                curr_basic_group_probs =\
                    [np.power(10.0, -elem) for elem in curr_basic_group_probs]
            probs_by_groups.append(curr_group_probs)
            basic_probs_by_groups.append(curr_basic_group_probs)
        return tags_by_groups, probs_by_groups, basic_probs_by_groups


    def _make_tag_embeddings(self, lm):
        # embeddings_weights.shape = (n_symbol_features, tag_embeddings_dim)
        embedding_weights = lm.get_embeddings_weights()
        if embedding_weights is None:
            return None
        # embeddings_weights.shape = (n_symbols, tag_embeddings_dim)
        embedding_weights = np.dot(lm.vocabulary_.symbol_matrix_, embedding_weights)
        self.tag_embeddings_dim_ = embedding_weights.shape[1]
        self.tag_embeddings_ = np.zeros(shape=(self.tags_number_, self.tag_embeddings_dim_))
        for i, tag in enumerate(self.tags_.symbols_):
            lm_index = lm.vocabulary_.toidx(tag)
            self.tag_embeddings_[i] = embedding_weights[lm_index]
        if self.normalize_lm_embeddings:
            self.tag_embeddings_ /= np.linalg.norm(self.tag_embeddings_, axis=1)[:,None]
        return self

    def build(self):
        word_inputs = kl.Input(shape=(None, MAX_WORD_LENGTH+2), dtype="int32")
        inputs = [word_inputs]
        if hasattr(self, "lm_"):
            if not self.use_fusion:
                lm_inputs = kl.Input(shape=(None, self.tags_number_), dtype="float32")
                inputs.append(lm_inputs)
            lm_state_inputs = kl.Input(shape=(None, self.lm_state_dim_), dtype="float32")
            inputs.append(lm_state_inputs)
        word_outputs = self.build_word_cnn(word_inputs)
        pre_outputs, lstm_outputs = self.build_basic_network(word_outputs)
        loss = (leader_loss(self.leader_loss_weight) if self.use_leader_loss
                else "categorical_crossentropy")
        compile_args = {"optimizer": ko.nadam(lr=0.002, clipnorm=5.0),
                        "loss": loss, "metrics": ["accuracy"]}
        if hasattr(self, "lm_"):
            position_inputs = kl.Lambda(positions_func)(word_inputs)
            if self.use_fusion:
                lm_state_inputs = TemporalDropout(lm_state_inputs, self.lm_dropout)
                fusion_inputs = kl.concatenate([lstm_outputs, lm_state_inputs, position_inputs])
                fusion_state_units = kl.TimeDistributed(
                    kl.Dense(self.fusion_state_units, activation="relu"))(fusion_inputs)
                final_outputs = kl.TimeDistributed(
                    kl.Dense(self.tags_number_, activation="softmax",
                             activity_regularizer=self.fusion_regularizer),
                    name = "p_output")(fusion_state_units)
                decoder_inputs = [pre_outputs, lstm_outputs, position_inputs, lm_state_inputs]
            else:
                if self.use_rnn_for_weight_state:
                    first_gate_inputs = kl.Bidirectional(kl.LSTM(
                        self.weight_state_rnn_units, dropout=self.lstm_dropout,
                        return_sequences=True))(word_outputs)
                else:
                    first_gate_inputs = word_outputs
                lm_inputs = TemporalDropout(lm_inputs, self.lm_dropout)
                gate_inputs = kl.concatenate([first_gate_inputs, lm_state_inputs, position_inputs])
                gate_layer = WeightedCombinationLayer(name="p_output",
                                                      first_threshold=self.probs_threshold,
                                                      second_threshold=self.lm_probs_threshold,
                                                      use_dimension_bias=self.use_dimension_bias,
                                                      use_intermediate_layer=self.use_intermediate_activation_for_weights,
                                                      intermediate_dim=self.intermediate_units_for_weights)
                final_outputs = gate_layer([pre_outputs, lm_inputs, gate_inputs])
                decoder_inputs = [pre_outputs, word_outputs, position_inputs, lm_inputs, lm_state_inputs]
            outputs = [final_outputs, pre_outputs]
            loss_weights = [1, self.base_model_weight]
            compile_args["loss_weights"] = loss_weights
        else:
            outputs = pre_outputs
        self.model_ = Model(inputs, outputs)
        self.model_.compile(**compile_args)
        if hasattr(self, "lm_"):
            self._basic_model_ = kb.Function([word_inputs, kb.learning_phase()], decoder_inputs[:3])
            self._decoder_ = kb.Function(decoder_inputs + [kb.learning_phase()], [final_outputs])
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
        # highway_input = kl.GlobalMaxPooling1D()(conv_output)
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
        if hasattr(self, "tag_embeddings_"):
            pre_outputs = self.tag_embeddings_output_layer(lstm_outputs)
        else:
            pre_outputs = kl.TimeDistributed(
                kl.Dense(self.tags_number_, activation="softmax",
                         activity_regularizer=self.regularizer),
                name="p")(lstm_outputs)
        return pre_outputs, lstm_outputs

    def tag_embeddings_output_layer(self, lstm_outputs):
        outputs_embeddings = kl.TimeDistributed(
            kl.Dense(self.tag_embeddings_dim_, use_bias=False))(lstm_outputs)
        if self.normalize_lm_embeddings:
            norm_layer = kl.Lambda(kb.l2_normalize, arguments={"axis": -1})
            outputs_embeddings = kl.TimeDistributed(norm_layer)(outputs_embeddings)
        score_layer = kl.Lambda(kb.dot, arguments={"y": kb.constant(self.tag_embeddings_.T)})
        scores = kl.TimeDistributed(score_layer)(outputs_embeddings)
        probs = kl.TimeDistributed(kl.Activation("softmax"), name="p")(scores)
        return probs


def extend_history(histories, hyps, indexes, start=0, pos=None,
                   history_pos=0, value=None, func="append", **kwargs):
    to_append = ([elem[pos] for elem in hyps]
                 if value is None else [value] * len(hyps))
    if func == "append":
        func = lambda x, y: x + [y]
    elif func == "sum":
        func = lambda x, y: x + y
    elif not callable(func):
        raise ValueError("func must be 'append', 'sum' or a callable object")
    group_histories = [func(histories[elem[history_pos]], value, **kwargs)
                       for elem, value in zip(hyps, to_append)]
    for i, index in enumerate(indexes):
        histories[start+index] = group_histories[i]






