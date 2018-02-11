"""
Эксперименты с языковыми моделями
"""
import sys
import os
import getopt
from collections import defaultdict
import bisect
import copy
import itertools
import inspect
import json
# import ujson as json

import numpy as np
np.set_printoptions(precision=3)

import keras
import keras.optimizers as ko
import keras.backend as kb
import keras.layers as kl
from keras.callbacks import EarlyStopping, ModelCheckpoint
#
from .common import AUXILIARY, BEGIN, END, UNKNOWN, PAD
from .common import repeat_
from .vocabulary import Vocabulary, vocabulary_from_json
from .cells import make_history, AttentionCell, AttentionCell3D
from .UD_preparation.read_tags import descr_to_feats

def to_one_hot(indices, num_classes):
    """
    Theano implementation for numpy arrays

    :param indices: np.array, dtype=int
    :param num_classes: int, число классов
    :return: answer, np.array, shape=indices.shape+(num_classes,)
    """
    shape = indices.shape
    indices = np.ravel(indices)
    answer = np.zeros(shape=(indices.shape[0], num_classes), dtype=int)
    answer[np.arange(indices.shape[0]), indices] = 1
    return answer.reshape(shape+(num_classes,))

def read_input(infile, label_field=None, max_num=-1):
    answer = []
    feats_column = 2 if label_field is None else 1
    with open(infile, "r", encoding="utf8") as fin:
        for i, line in enumerate(fin):
            if i == max_num:
                break
            line = line.strip()
            if line == "":
                continue
            splitted = line.split()
            curr_elem = [splitted[1]]
            feats = splitted[feats_column] if len(splitted) > feats_column else ""
            feats = [x.split("=") for x in feats.split(",")]
            feats = {x[0]: x[1] for x in feats}
            if label_field is not None:
                label = feats.pop(label_field, None)
            else:
                label = splitted[1] if len(splitted) > 1 else None
            if label is not None:
                curr_elem.append(label)
                curr_elem.append(feats)
            answer.append(curr_elem)
    return answer


def make_bucket_indexes(lengths, buckets_number=None,
                        bucket_size=None, join_buckets=True):
    if buckets_number is None and bucket_size is None:
        raise ValueError("Either buckets_number or bucket_size should be given")
    indexes = np.argsort(lengths)
    lengths = sorted(lengths)
    m = len(lengths)
    if buckets_number is not None:
        level_indexes = [m * (i+1) // buckets_number for i in range(buckets_number)]
    else:
        level_indexes = [min(start+bucket_size, m) for start in range(0, m, bucket_size)]
    if join_buckets:
        new_level_indexes = []
        for i, index in enumerate(level_indexes[:-1]):
            if lengths[index-1] < lengths[level_indexes[i+1]-1]:
                new_level_indexes.append(index)
        level_indexes = new_level_indexes + [m]
    bucket_indexes =  [indexes[start:end] for start, end in
                       zip([0] + level_indexes[:-1], level_indexes)]
    bucket_lengths = [lengths[i-1] for i in level_indexes]
    return bucket_indexes, bucket_lengths

class NeuralLM:

    def __init__(self, reverse=False, symbols_has_features=False,
                 use_label=False, use_feats=False,
                 history=1, use_attention=False, use_output_rnn=True,
                 attention_activation="concatenate",
                 use_attention_bias=False, memory_embeddings_size=8,
                 use_embeddings=False, embeddings_size=8,
                 use_feature_embeddings=False, feature_embeddings_size=8,
                 rnn="lstm", rnn_size=16, dropout=0.2,
                 batch_size=32, nepochs=20, validation_split=0.2,
                 random_state=187, verbose=1, callbacks=None):
        self.reverse = reverse
        self.symbols_has_features = symbols_has_features
        self.use_label = use_label
        self.use_feats = use_feats
        self.history = history
        self.use_attention = use_attention
        self.use_output_rnn = use_output_rnn
        self.attention_activation = attention_activation
        self.use_attention_bias = use_attention_bias
        self.memory_embeddings_size = memory_embeddings_size
        self.use_embeddings = use_embeddings
        self.embeddings_size = embeddings_size
        self.use_feature_embeddings = use_feature_embeddings
        self.feature_embeddings_size = feature_embeddings_size
        self.rnn = rnn
        self.rnn_size = rnn_size
        self.dropout = dropout
        self.batch_size = batch_size
        self.nepochs = nepochs
        self.validation_split = validation_split
        self.random_state = random_state
        self.verbose = verbose
        self.callbacks = callbacks if callbacks is not None else []

    def to_json(self, outfile, model_file):
        info = dict()
        model_file = os.path.abspath(model_file)
        for (attr, val) in inspect.getmembers(self):
            if not (attr.startswith("__") or inspect.ismethod(val) or
                    isinstance(val, property) or
                    isinstance(val, Vocabulary) or
                    attr.isupper() or attr in ["callbacks", "model_", "_attention_func_"]):
                info[attr] = val
            elif isinstance(val, Vocabulary):
                info[attr] = val.jsonize()
            elif attr == "model_":
                info["dump_file"] = model_file
                self.model_.save_weights(model_file)
            elif attr == "callbacks":
                for callback in val:
                    if isinstance(callback, EarlyStopping):
                        info["early_stopping_callback"] = {"patience": callback.patience,
                                                           "monitor": callback.monitor}
                    # elif isinstance(callback, ReduceLROnPlateau):
                    #     curr_val = dict()
                    #     for key in ["patience", "monitor", "factor"]:
                    #         curr_val[key] = getattr(callback, key)
                    #     info["reduce_LR_callback"] = curr_val
                    elif isinstance(callback, ModelCheckpoint):
                        info["model_checkpoint_callback"] =\
                            {key: getattr(callback, key) for key in ["monitor", "filepath"]}
        with open(outfile, "w", encoding="utf8") as fout:
            json.dump(info, fout)

    def make_vocabulary(self, X):
        """
        :param X: list of lists,
            создаёт словарь на корпусе X = [x_1, ..., x_m]
            x_i = [w_i, (c_i, (feats_i)) ], где
                w: str, строка из корпуса
                c: str(optional), класс строки (например, часть речи)
                feats: dict(optional), feats = {f_1: v_1, ..., f_k: v_k},
                    набор пар признак-значение (например, значения грамматических категорий)
        :return: self, обученная модель
        """
        # первый проход: определяем длины строк, извлекаем словарь символов и признаков
        symbols, labels, features = set(), set(), defaultdict(set)
        for elem in X:
            word = elem[0]
            label = elem[1] if len(elem) > 1 else None
            feats = elem[2] if len(elem) > 2 else None
            symbols.update(word)
            if self.use_label and label is not None:
                labels.add(label)
                if self.use_feats and feats is not None:
                    for feature, value in feats.items():
                        features[label + "_" + feature].add(value)
        # создаём словари нужного размера
        self.symbols_ = AUXILIARY + sorted(symbols)
        self.symbol_codes_ = {x: i for i, x in enumerate(self.symbols_)}
        if len(labels) > 0:
            self.labels_ = AUXILIARY + sorted(labels)
            self.label_codes_ = {x: i for i, x in enumerate(self.labels_)}
        else:
            self.labels_, self.label_codes_ = None, None
        self.feature_values_, self.feature_codes_ = [], dict()
        for i, (feat, values) in enumerate(sorted(features.items())):
            self.feature_values_.append({value: j for j, value in enumerate(values)})
            self.feature_codes_[feat] = i
        self.feature_offsets_ = np.concatenate(
            ([0], np.cumsum([len(x) for x in self.feature_values_], dtype=np.int32)))
        self.feature_offsets_ = [int(x) for x in self.feature_offsets_]
        print("Symbols: {}, labels: {}, feature values: {}".format(
            len(self.symbols_), self.labels_number, self.feature_offsets_[-1]))
        return self

    def _make_vocabulary(self, X):
        self.vocabulary_ = Vocabulary()
        self.vocabulary_.train((elem[0] for elem in X))
        return self

    def make_symbol_features_vocabulary(self, X):
        """
        Constructs symbol features vocabulary when
        symbols are not atomic but consist of feature-values pairs
        """
        labels, features = set(), defaultdict(set)
        for elem in X:
            word = elem[0]
            for symbol in word:
                symbol, feats = descr_to_feats(symbol)
                labels.add(symbol)
                for feature, value in feats:
                    labels.add("{}_{}_{}".format(symbol, feature, value))
        # создаём словари нужного размера
        labels = sorted(labels, key=(lambda x: (x.count("_"), x)))
        self.symbol_labels_ = AUXILIARY + labels
        self.symbol_labels_codes_ = {x: i for i, x in enumerate(self.symbol_labels_)}
        self.symbol_vector_size_ = len(self.symbol_labels_)
        symbol_labels_count = sum(int("_" not in x) for x in labels)
        print("Symbol labels: {}, symbol feature values: {}".format(
            symbol_labels_count, len(self.symbol_labels_)))
        return self

    @property
    def input_symbols_number(self):
        # return self.symbol_vector_size_ if self.symbols_has_features else len(self.symbols_)
        return self.vocabulary_.symbols_number_

    @property
    def output_symbols_number(self):
        # return len(self.symbols_)
        return self.vocabulary_.symbols_number_

    @property
    def labels_number(self):
        return len(self.labels_) if self.labels_ is not None else 0

    @property
    def feature_vector_size(self):
        return self.labels_number + self.feature_offsets_[-1]

    def _make_word_vector(self, word, bucket_length=None, symbols_has_features=False):
        """
        :param word:
        :param pad:
        :return:
        """
        m = len(word)
        if bucket_length is None:
            bucket_length = m + 2
        if symbols_has_features:
            answer = np.zeros(shape=(bucket_length, self.symbol_vector_size_), dtype=np.uint8)
            answer[0, BEGIN], answer[m+1, END] = 1, 1
            answer[m+2:, PAD] = 1
            for i, x in enumerate(word, 1):
                x = descr_to_feats(x)
                answer[i, self._get_symbol_features_codes(*x)] = 1
        else:
            answer = np.full(shape=(bucket_length,), fill_value=PAD, dtype=np.uint8)
            answer[0], answer[m+1] = BEGIN, END
            for i, x in enumerate(word, 1):
                answer[i] = self.symbol_codes_.get(x, UNKNOWN)
        return answer

    def _make_feature_vector(self, label, feats):
        answer = np.zeros(shape=(self.feature_vector_size,))
        label_code = self.label_codes_.get(label, UNKNOWN)
        answer[label_code] = 1
        if label_code != UNKNOWN:
            # класс известен, поэтому есть смысл рассматривать признаки
            for feature, value in feats.items():
                feature = label + "_" + feature
                feature_code = self.feature_codes_.get(feature)
                if self.use_feats and feature_code is not None:
                    value_code = self.feature_values_[feature_code].get(value)
                    if value_code is not None:
                        value_code += self.feature_offsets_[feature_code]
                        answer[value_code + self.labels_number] = 1
        return answer

    def _get_symbol_features_codes(self, symbol, feats):
        symbol_code = self.symbol_labels_codes_.get(symbol, UNKNOWN)
        answer = [symbol_code]
        if symbol_code == UNKNOWN:
            return answer
        for feature, value in feats:
            feature_repr = "{}_{}_{}".format(symbol, feature, value)
            symbol_code = self.symbol_labels_codes_.get(feature_repr)
            if symbol_code is not None:
                answer.append(symbol_code)
        return answer

    def transform(self, X, pad=True, return_indexes=True,
                  buckets_number=None, bucket_size=None, join_buckets=True):
        lengths = [len(x[0])+2 for x in X]
        if pad:
            indexes, level_lengths = make_bucket_indexes(
                lengths, buckets_number=buckets_number,
                bucket_size=bucket_size, join_buckets=join_buckets)
        else:
            indexes = [[i] for i in range(len(X))]
            level_lengths = lengths
        answer = [None] * len(X)
        for bucket_indexes, bucket_length in zip(indexes, level_lengths):
            for i in bucket_indexes:
                elem = X[i]
                word = elem[0]
                if self.reverse:
                    word = word[::-1]
                label = elem[1] if len(elem) > 1 else None
                feats = elem[2] if len(elem) > 2 else None
                word_vector = self._make_word_vector(word, bucket_length=bucket_length,
                                                     symbols_has_features=self.symbols_has_features)
                if self.labels_ is not None:
                    feature_vector = self._make_feature_vector(label, feats)
                    to_append = [word_vector, feature_vector]
                else:
                    to_append = [word_vector]
                if self.symbols_has_features:
                    word_vector = self._make_word_vector(word, bucket_length=bucket_length)
                    to_append.append(word_vector)
                answer[i] = to_append
        if return_indexes:
            return answer, indexes
        else:
            return answer

    def train(self, X, X_dev=None, model_file=None):
        np.random.seed(self.random_state)  # initialize the random number generator
        # self.make_vocabulary(X)
        # if self.symbols_has_features:
        #     self.make_symbol_features_vocabulary(X)
        self._make_vocabulary()
        X_train, indexes_by_buckets = self.transform(X, buckets_number=10)
        if X_dev is not None:
            X_dev, dev_indexes_by_buckets = self.transform(X_dev, bucket_size=256, join_buckets=False)
        else:
            X_dev, dev_indexes_by_buckets = None, None
        self.build()
        self.train_model(X_train, indexes_by_buckets, X_dev,
                         dev_indexes_by_buckets, model_file=model_file)
        return self

    def build(self):
        RNN = kl.GRU if self.rnn == "gru" else kl.LSTM
        if self.symbols_has_features:
            symbol_inputs = kl.Input(shape=(None, self.input_symbols_number), dtype='int32')
        else:
            symbol_inputs = kl.Input(shape=(None,), dtype='int32')
        symbol_embeddings, symbol_inputs_length = self._build_symbol_layer(symbol_inputs)
        if not self.use_attention:
            if self.history > 1:
                symbol_inputs_length *= self.history
                pad = kb.zeros_like(symbol_embeddings[0,0])
                symbol_embeddings = kl.Lambda(
                    make_history, arguments={"h": self.history, "pad": pad, "flatten": True},
                    output_shape=(None, symbol_inputs_length))(symbol_embeddings)
            to_concatenate = [symbol_embeddings]
        else:
            encodings = RNN(self.memory_embeddings_size,
                            return_sequences=True)(symbol_embeddings)
            if self.dropout > 0.0:
                encodings = kl.Dropout(self.dropout)(encodings)
            if self.symbols_has_features or self.use_embeddings:
                AttentionLayer, attention_inputs = AttentionCell3D, symbol_embeddings
            else:
                AttentionLayer, attention_inputs = AttentionCell, symbol_inputs
            attention_layer = AttentionLayer(self.history, symbol_inputs_length,
                                             self.memory_embeddings_size,
                                             use_bias=self.use_attention_bias)
            memory, attention_probs = attention_layer([attention_inputs, encodings])
            to_concatenate = [memory]
        if self.labels_ is not None:
            feature_inputs = kl.Input(shape=(self.feature_vector_size,))
            inputs = [symbol_inputs, feature_inputs]
            feature_inputs_length = self.feature_vector_size
            if self.use_feature_embeddings:
                feature_inputs = kl.Dense(self.feature_embeddings_size,
                                          input_shape=(self.feature_vector_size,),
                                          activation="relu",  use_bias=False)(feature_inputs)
                feature_inputs_length = self.feature_embeddings_size
            # cannot use kb.repeat_elements because it requires an integer
            feature_inputs = kl.Lambda(
                repeat_, arguments={"k": kb.shape(symbol_embeddings)[1]},
                output_shape=(None, feature_inputs_length))(feature_inputs)
            to_concatenate.append(feature_inputs)
        else:
            inputs = [symbol_inputs]
        lstm_inputs = (kl.Concatenate()(to_concatenate) if len(to_concatenate) > 1
                       else to_concatenate[0])
        if not self.use_attention or self.use_output_rnn:
            lstm_outputs = RNN(self.rnn_size, return_sequences=True)(lstm_inputs)
            if self.dropout > 0.0:
                lstm_outputs = kl.Dropout(self.dropout)(lstm_outputs)
        else:
            # no LSTM over memory blocks
            lstm_outputs = lstm_inputs
        outputs = kl.TimeDistributed(
            kl.Dense(self.output_symbols_number, activation="softmax",
                     input_shape=(self.rnn_size,)), name="output")(lstm_outputs)
        compile_args = {"optimizer": ko.nadam(clipnorm=5.0), "loss": "categorical_crossentropy"}
        self.model_ = keras.models.Model(inputs, outputs)
        self.model_.compile(**compile_args)
        if self.verbose > 0:
            print(self.model_.summary())
        if self.use_attention:
            self._attention_func_ = kb.Function(
                inputs + [kb.learning_phase()], [attention_probs])
        return self

    def _build_symbol_layer(self, symbol_inputs):
        if self.symbols_has_features:
            symbol_inputs = kl.Lambda(kb.cast,
                                      arguments={"dtype": "float32"},
                                      output_shape=(lambda x: x))(symbol_inputs)
        if self.use_embeddings:
            if self.symbols_has_features:
                symbol_embeddings = kl.TimeDistributed(
                    kl.Dense(self.embeddings_size, use_bias=False))(symbol_inputs)
            else:
                symbol_embeddings = kl.Embedding(
                    self.input_symbols_number, self.embeddings_size)(symbol_inputs)
            if self.dropout > 0.0:
                symbol_embeddings = kl.Dropout(self.dropout)(symbol_embeddings)
            symbol_inputs_length = self.embeddings_size
        else:
            if not self.symbols_has_features:
                embedding = kl.Lambda(kb.one_hot,
                                      arguments={"num_classes": self.input_symbols_number},
                                      output_shape=(None, self.input_symbols_number))
                symbol_embeddings = embedding(symbol_inputs)
            else:
                symbol_embeddings = symbol_inputs
            symbol_inputs_length = self.input_symbols_number
        return symbol_embeddings, symbol_inputs_length

    def train_model(self, X, indexes_by_buckets,
                    X_dev=None, dev_indexes_by_buckets=None, model_file=None):
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
        # train_batches_indexes = list(itertools.chain.from_iterable(
        #     (((i, j) for j in range(0, len(bucket), self.batch_size))
        #      for i, bucket in enumerate(train_indexes_by_buckets))))
        # dev_batches_indexes = list(itertools.chain.from_iterable(
        #     (((i, j) for j in range(0, len(bucket), 256))
        #      for i, bucket in enumerate(dev_indexes_by_buckets))))
        if model_file is not None:
            callback = ModelCheckpoint(model_file, save_weights_only=True,
                                       save_best_only=True)
            if self.callbacks is not None:
                self.callbacks.append(callback)
            else:
                self.callbacks = [callback]
        train_steps = sum((1 + (len(x)-1) // self.batch_size) for x in train_indexes_by_buckets)
        dev_steps = len(dev_indexes_by_buckets)
        train_gen = self._generate_data(X, train_indexes_by_buckets, self.batch_size)
        dev_gen = self._generate_data(X_dev, dev_indexes_by_buckets, shuffle=False)
        self.model_.fit_generator(
            train_gen, steps_per_epoch=train_steps, epochs=self.nepochs,
            callbacks=self.callbacks, validation_data=dev_gen,
            validation_steps=dev_steps, verbose=1)
        if model_file is not None:
            self.model_.load_weights(model_file)
        return self

    def _generate_data(self, X, indexes_by_buckets, batch_size=None,
                       shuffle=True, yield_weights=True):
        fields_number = 2 if self.labels_ is not None else 1
        answer_index = -1 if self.symbols_has_features else 0
        if batch_size is None:
            batches_indexes = [(i, 0) for i in range(len(indexes_by_buckets))]
        else:
            batches_indexes = list(itertools.chain.from_iterable(
                (((i, j) for j in range(0, len(bucket), batch_size))
                 for i, bucket in enumerate(indexes_by_buckets))))
        # total_data_size = sum(np.count_nonzero(X[j][answer_index] != PAD)-1
        #                       for elem in indexes_by_buckets for j in elem)
        # total_arrays_size = sum(len(X[j][answer_index])
        #                         for elem in indexes_by_buckets for j in elem)
        total_arrays_size = sum(np.count_nonzero(X[j][answer_index] != PAD) - 1
                                for elem in indexes_by_buckets for j in elem)
        total_data_length = sum(len(elem) for elem in indexes_by_buckets)
        S, L = 0, 0
        while True:
            if shuffle:
                for elem in indexes_by_buckets:
                    np.random.shuffle(elem)
                np.random.shuffle(batches_indexes)
            for i, start in batches_indexes:
                bucket_size = len(indexes_by_buckets[i])
                end = min(bucket_size, start + batch_size) if batch_size is not None else bucket_size
                bucket_indexes = indexes_by_buckets[i][start:end]
                to_yield = [np.array([X[j][k] for j in bucket_indexes])
                            for k in range(fields_number)]
                indexes_to_yield = np.array([X[j][answer_index] for j in bucket_indexes])
                padding = np.full(shape=(end - start, 1), fill_value=PAD)
                indexes_to_yield = np.hstack((indexes_to_yield[:,1:], padding))
                y_to_yield = to_one_hot(indexes_to_yield, self.output_symbols_number)
                weights_to_yield = np.ones(shape=(end - start,), dtype=np.float32)
                # array_size = np.sum(np.ones_like(indexes_to_yield))
                # array_size = np.count_nonzero(indexes_to_yield != PAD)
                if yield_weights:
                    weights_to_yield *= total_data_length * indexes_to_yield.shape[1]
                    weights_to_yield /= total_arrays_size
                yield (to_yield, y_to_yield, weights_to_yield)

    def _score_batch(self, bucket, answer, batch_size=1):
        """
        :Arguments
         batch: list of np.arrays, [data, (features)]
            data: shape=(batch_size, length)
            features(optional): shape=(batch_size, self.feature_vector_size)
        :return:
        """
        # elem[0] because elem = [word, (pos, (feats))]
        # ПАДДИНГ ПОПАДАЕТ В УЧИТЫВАЕМЫЕ СИМВОЛЫ
        bucket_size, length = bucket[0].shape[:2]
        padding = np.full(answer[:,:1].shape, PAD, answer.dtype)
        shifted_data = np.hstack((answer[:,1:], padding))
        mask = (shifted_data != PAD)
        lengths = np.argmin(mask, axis=1)
        # evaluate принимает только данные того же формата, что и выход модели
        answers = to_one_hot(shifted_data, self.output_symbols_number)
        # total = self.model.evaluate(bucket, answers, batch_size=batch_size)
        # last two scores are probabilities of word end and final padding symbol
        scores = self.model_.predict(bucket, batch_size=batch_size)
        answers_, scores_ = kb.constant(answers), kb.constant(scores)
        # using 2-based logarithm
        losses = kb.eval(kb.categorical_crossentropy(answers_, scores_))
        # total = [np.sum(elem[:length]) for elem, length in zip(losses, lengths)] # / np.log(2.0)
        total = np.sum(losses, axis=1) # / np.log(2.0)
        letter_scores = scores[np.arange(bucket_size)[:,np.newaxis],
                               np.arange(length)[np.newaxis,:], shifted_data]
        letter_scores = [elem[:length] for elem, length in zip(letter_scores, lengths)]
        return letter_scores, total

    def score(self, x, **args):
        return self.predict([x], batch_size=1, **args)

    def predict(self, X, batch_size=32, return_letter_scores=False,
                return_log_probs=False, return_exp_total=False):
        """
        Вычисляет логарифмические вероятности для всех слов в X,
        а также вероятности отдельных символов
        """
        # сделать batch_size и подавать вход по блокам
        fields_number = 2 if self.labels_ is not None else 1
        answer_index = -1 if self.symbols_has_features else 0
        X_test, indexes = self.transform(X, bucket_size=batch_size, join_buckets=False)
        answer = [None] * len(X)
        for curr_indexes in indexes:
            X_curr = [np.array([X_test[j][k] for j in curr_indexes]) for k in range(fields_number)]
            y_curr = np.array([X_test[j][answer_index] for j in curr_indexes])
            letter_scores, total_scores = self._score_batch(X_curr, y_curr, batch_size=batch_size)
            if return_log_probs:
                # letter_scores = -np.log2(letter_scores)
                letter_scores = [-np.log(letter_score) for letter_score in letter_scores]
            if return_exp_total:
                total_scores = 2.0 ** total_scores
            for i, letter_score, total_score in zip(curr_indexes, letter_scores, total_scores):
                answer[i] = (letter_score, total_score) if return_letter_scores else total_score
        return answer

    def predict_proba(self, X, batch_size=256):
        fields_number = 2 if self.labels_ is not None else 1
        X_test, indexes = self.transform(X, pad=False, return_indexes=True)
        answer = [None] * len(X)
        start_probs = np.zeros(shape=(1, self.output_symbols_number), dtype=float)
        start_probs[0, BEGIN] = 1.0
        for curr_indexes in indexes:
            X_curr = [np.array([X_test[j][k] for j in curr_indexes]) for k in range(fields_number)]
            curr_probs = self.model_.predict(X_curr, batch_size=batch_size)
            for i, probs in zip(curr_indexes, curr_probs):
                answer[i] = np.vstack((start_probs, probs[:len(X[i][0])+1]))
        return answer

    def predict_attention(self, X, batch_size=32):
        fields_number = 2 if self.labels_ is not None else 1
        X_test, indexes = self.transform(X, pad=False, return_indexes=True)
        answer = [None] * len(X)
        for curr_indexes in indexes:
            X_curr = [np.array([X_test[j][k] for j in curr_indexes]) for k in range(fields_number)]
            # нужно добавить фазу обучения (используется dropout)
            curr_attention = self._attention_func_(X_curr + [0])
            for i, elem in zip(curr_indexes, curr_attention[0]):
                answer[i] = elem
        return answer


    def perplexity(self, X, bucket_size=None):
        # возможно, нужно переписать на batches
        X_test, indexes = self.transform(X, bucket_size=bucket_size, join_buckets=False)
        # print(indexes)
        # if self.symbols_has_features:
        #     answer = [x[-1] for x in X_test]
        #     X_test = [x[:-1] for x in X_test]
        # else:
        #     answer = [x[0] for x in X_test]
        # for i, elem in enumerate(answer):
        #     padding = np.full(elem[:,:1].shape, PAD)
        #     answer[i] = np.concatenate([elem[:,1:], padding])
        loss = self.model_.evaluate_generator(
            self._generate_data(X_test, indexes, batch_size=bucket_size,
                                shuffle=False), len(indexes))
        # loss /= np.log(2.0)
        # total_items, active_items = 0, 0
        # for elem in X_test:
        #     total_items += np.sum(np.ones_like(elem[1], dtype=int))
        #     active_items += np.sum((elem[1] > BEGIN).astype(int))
        # print(active_items, total_items)
        # loss /= (active_items / total_items)
        return loss

def load_lm(infile):
    with open(infile, "r", encoding="utf8") as fin:
        json_data = json.load(fin)
    args = {key: value for key, value in json_data.items()
            if not (key.endswith("_") or key.endswith("callback") or key.endswith("dump_file"))}
    callbacks = []
    early_stopping_callback_data = json_data.get("early_stopping_callback")
    if early_stopping_callback_data is not None:
        callbacks.append(EarlyStopping(**early_stopping_callback_data))
    # reduce_LR_callback_data = json_data.get("reduce_LR_callback")
    # if reduce_LR_callback_data is not None:
    #     callbacks.append(ReduceLROnPlateau(**reduce_LR_callback_data))
    model_checkpoint_callback_data = json_data.get("model_checkpoint_callback")
    if model_checkpoint_callback_data is not None:
        callbacks.append(ModelCheckpoint(**model_checkpoint_callback_data))
    args['callbacks'] = callbacks
    # создаём языковую модель
    lm = NeuralLM(**args)
    # обучаемые параметры
    args = {key: value for key, value in json_data.items() if key[-1] == "_"}
    for key, value in args.items():
        if key == "vocabulary_":
            setattr(lm, key, vocabulary_from_json(value))
        setattr(lm, key, value)
    # модель
    lm.build()  # не работает сохранение модели, приходится сохранять только веса
    lm.model_.load_weights(json_data['dump_file'])
    return lm


