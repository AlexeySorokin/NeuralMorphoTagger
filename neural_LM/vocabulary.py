from collections import defaultdict
import inspect


from .common import *
from .UD_preparation.read_tags import descr_to_feats


def vocabulary_from_json(info):
    vocab = Vocabulary(info["character"])
    for attr, val in info.items():
        print(attr)
        setattr(vocab, attr, val)
    return vocab

class Vocabulary:

    def __init__(self, character=False):
        self.character = character

    def train(self, text):
        symbols = set()
        for elem in text:
            if self.character:
                for x in elem:
                    symbols.update(x)
            else:
                symbols.update(elem)
        self.symbols_ = AUXILIARY + sorted(symbols)
        self.symbol_codes_ = {x: i for i, x in enumerate(self.symbols_)}
        return self

    @property
    def symbols_number_(self):
        return len(self.symbols_)

    def jsonize(self):
        info = {attr: val for attr, val in inspect.getmembers(self)
                if not(attr.startswith("__") or inspect.ismethod(val)
                or isinstance(getattr(Vocabulary, attr, None), property))}
        return info



class FeatureVocabulary(Vocabulary):

    def __init__(self, character=False):
        super().__init__(character=character)

    def train(self, text):
        super().train(text)
        labels, features = set(), defaultdict(set)
        for word in text:
            if self.character:
                symbols = {x for symbol in word for x in symbol}
            else:
                symbols = set(word)
            for symbol in symbols:
                symbol, feats = descr_to_feats(symbol)
                labels.add(symbol)
                for feature, value in feats:
                    labels.add("{}_{}_{}".format(symbol, feature, value))
        # создаём словари нужного размера
        labels = sorted(labels, key=(lambda x: (x.count("_"), x)))
        self.symbol_labels_ = AUXILIARY + labels
        self.symbol_labels_codes_ = {x: i for i, x in enumerate(self.symbol_labels_)}
        return self

    @property
    def symbol_vector_size_(self):
        return len(self.symbol_labels_)


