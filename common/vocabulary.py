from collections import defaultdict
import inspect


from .common import *
from .UD_preparation import descr_to_feats


def vocabulary_from_json(info, use_features=False):
    cls = FeatureVocabulary if use_features else Vocabulary
    info_to_initialize = dict(elem for elem in info.items() if elem[0][-1] != "_")
    vocab = cls(**info_to_initialize)
    args = dict()
    for attr, val in info.items():
        # if attr[-1] == "_" and not isinstance(getattr(cls, attr, None), property):
        if attr in ["symbols_", "symbol_codes_"]:
            setattr(vocab, attr, val)
    if hasattr(vocab, "symbols_") and not hasattr(vocab, "symbol_codes_"):
        vocab.symbol_codes_ = {x: i for i, x in enumerate(vocab.symbols_)}
    if use_features:
        vocab._make_features(**args)
    return vocab

class Vocabulary:

    def __init__(self, character=False, min_count=1):
        self.character = character
        self.min_count = min_count

    def train(self, text):
        symbols = defaultdict(int)
        for elem in text:
            if self.character:
                curr_symbols = [symbol for x in elem for symbol in x]
            else:
                curr_symbols = elem
            for x in curr_symbols:
                symbols[x] += 1
        symbols = [x for x, count in symbols.items() if count >= self.min_count]
        self.symbols_ = AUXILIARY + sorted(symbols)
        self.symbol_codes_ = {x: i for i, x in enumerate(self.symbols_)}
        return self

    def toidx(self, x):
        return self.symbol_codes_.get(x, UNKNOWN)

    @property
    def symbols_number_(self):
        return len(self.symbols_)

    def jsonize(self):
        info = {attr: val for attr, val in inspect.getmembers(self)
                if (not(attr.startswith("__") or inspect.ismethod(val))
                    and (attr[-1] != "_" or attr in ["symbols_", "symbol_codes_"]))}
        return info


class FeatureVocabulary(Vocabulary):

    def __init__(self, character=False, min_count=1):
        super().__init__(character=character, min_count=min_count)

    def train(self, text):
        super().train(text)
        self._make_features()
        return self

    def _make_features(self):
        labels = set()
        # first pass determines the set of feature-value pairs
        for symbol in self.symbols_[4:]:
            symbol, feats = descr_to_feats(symbol)
            labels.add(symbol)
            for feature, value in feats:
                labels.add("{}_{}_{}".format(symbol, feature, value))
        labels = sorted(labels, key=(lambda x: (x.count("_"), x)))
        self.symbol_labels_ = AUXILIARY + labels
        self.symbol_labels_codes_ = {x: i for i, x in enumerate(self.symbol_labels_)}
        # second pass: constructing symbol-feature matrix
        self.symbol_matrix_ = np.zeros(shape=(len(self.symbols_), len(self.symbol_labels_)))
        for i, symbol in enumerate(self.symbols_):
            if symbol in AUXILIARY:
                codes = [i]
            else:
                symbol, feats = descr_to_feats(symbol)
                curr_labels = {symbol} | {"{}_{}_{}".format(symbol, *x) for x in feats}
                codes = [self.symbol_labels_codes_[label] for label in curr_labels]
            self.symbol_matrix_[i, codes] = 1
        return self

    def get_feature_code(self, x):
        return self.symbol_labels_codes_.get(x, UNKNOWN)

    @property
    def symbol_vector_size_(self):
        return len(self.symbol_labels_)

    def jsonize(self):
        return super().jsonize()


