import sys
import getopt
import json

import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from neural_LM.neural_lm import NeuralLM, read_input, load_lm
from neural_LM.UD_preparation.read_tags import read_tags_input
from neural_LM.UD_preparation.extract_tags_from_UD import read_tags_infile, extract_frequent_words


DEFAULT_NONE_PARAMS = ["model_file", "test_file", "outfile", "attention_file",
                       "train_files", "dev_files", "dump_file", "save_file",
                       "detailed_outfile"]
DEFAULT_PARAMS = {"symbols_has_features": False, "symbols_has_tokens": False}
BUCKET_SIZE = 256

def read_config(infile):
    with open(infile, "r", encoding="utf8") as fin:
        from_json = json.load(fin)
    params = dict()
    for param in DEFAULT_NONE_PARAMS:
        params[param] = from_json.get(param)
    for param, default_value in DEFAULT_PARAMS.items():
        params[param] = from_json.get(param, default_value)
    for param, value in from_json.items():
        if param not in params:
            params[param] = value
    if "model_params" not in params:
        params["model_params"] = dict()
    params["model_params"]["symbols_has_features"] = params["symbols_has_features"]
    params["model_params"]["symbols_has_tokens"] = params["symbols_has_tokens"]
    return params


def output_results(outfile, test_data, predictions, reverse=False,
                   adjust_boundary=True, on_separate_lines=False,
                   symbols=None, probs=None, best_symbols_number=5):
    if probs is None:
        probs = [None] * len(test_data)
    with open(outfile, "w", encoding="utf8") as fout:
        for elem, (symbol_scores, word_score), word_probs in\
                zip(test_data, predictions, probs):
            if reverse:
                symbol_scores = symbol_scores[::-1]
            word = list(elem[0])
            label = elem[1] if len(elem) > 1 else None
            # features = elem[2] if len(elem) > 2 else None
            if adjust_boundary:
                word = ["^"] + word if lm.reverse else (word + ["$"])
            if on_separate_lines:
                # if word elements are not elementary symbols
                fout.write((" " if on_separate_lines else "").join(word))
                if label is not None:
                    fout.write("\t" + tag)
                fout.write("\n{:.2f}\n".format(word_score))
                for i, x in enumerate(zip(word, symbol_scores)):
                    fout.write("{}\t{:.3f}\n".format(*x))
                    if word_probs is not None:
                        curr_probs = -np.log(np.clip(word_probs[i+1], 1e-16, 1))
                        best_indexes = np.argsort(curr_probs)[:best_symbols_number]
                        best_probs = [curr_probs[j] for j in best_indexes]
                        best_symbols = [symbols[j] for j in best_indexes]
                        fout.write(" ".join("{}-{:.3f}".format(*y)
                                            for y in zip(best_symbols, best_probs)))
                        fout.write("\n")
                fout.write("\n")
            elif word_probs is None:
                fout.write("{}\t{:.2f}\n{}\n".format(
                    word, word_score,
                    " ".join("{}-{:.3f}".format(x, y)
                             for x, y in zip(adjusted_word + "$", symbol_scores))))
            else:
                raise NotImplementedError


if __name__ == '__main__':
    if len(sys.argv[1:]) != 1:
        sys.exit("Usage: main.py <config json file>")
    params = read_config(sys.argv[1])
    callbacks = []
    if "stop_callback" in params:
        stop_callback = EarlyStopping(**params["stop_callback"])
        callbacks.append(stop_callback)
    if "LR_callback" in params:
        lr_callback = ReduceLROnPlateau(**params["LR_callback"])
        callbacks.append(lr_callback)
    if len(callbacks) == 0:
        callbacks = None
    params["model_params"]["callbacks"] = callbacks
    symbols_has_features = params["symbols_has_features"]
    symbols_has_tokens = params.get("symbols_has_tokens", False)
    token_reading_params = params.get("token_reading_params", dict())
    if params["train_files"] is not None:
        # maximal history of 5 symbols is effective
        lm = NeuralLM(**params["model_params"])
        train_data = []
        if symbols_has_tokens:
            frequent_pairs = extract_frequent_words(
                params["train_files"], **token_reading_params)
            token_reading_params = dict((key, token_reading_params[key])
                                        for key in ["to_lower", "append_case"]
                                        if key in token_reading_params)
        else:
            frequent_pairs, token_reading_params = None, dict()
        if "max_sents" in params:
            token_reading_params["max_sents"] = params["max_sents"]
        for train_file in params["train_files"]:
            if symbols_has_features:
                train_data += read_tags_infile(train_file, wrap=True,
                                               attach_tokens=symbols_has_tokens,
                                               **token_reading_params)
            else:
                train_data += read_input(train_file, label_field="pos")
            print(len(train_data))
        if params["dev_files"] is not None:
            dev_data = []
            if "max_sents" in params:
                del token_reading_params["max_sents"]
            for dev_file in params["dev_files"]:
                if symbols_has_features:
                    dev_data += read_tags_infile(dev_file, wrap=True,
                                                 attach_tokens=symbols_has_tokens,
                                                 **token_reading_params)
                else:
                    dev_data += read_input(dev_file, label_field="pos")
        else:
            dev_data = None
        lm.train(train_data, X_dev=dev_data, frequent_tokens=frequent_pairs,
                 model_file=params["model_file"], save_file=params["save_file"])
    elif params["load_file"] is not None:
        lm, train_data = load_lm(params["load_file"]), None
    else:
        raise ValueError("Either train_file or load_file should be given")
    if params["save_file"] is not None and params["dump_file"] is not None:
        lm.to_json(params["save_file"], params["dump_file"])
    if params["attention_file"] is not None:
        if train_data is None:
            print("Impossible to output attention: no training data")
        attention_scores = lm.predict_attention(train_data, batch_size=128)
        with open(params["attention_file"], "w", encoding="utf8") as fout:
            for (word, tag, feats), curr_attention in zip(train_data, attention_scores):
                if lm.reverse:
                    # слова читались с конца
                    curr_attention = curr_attention[::-1]
                adjusted_word = ("^" + word) if lm.reverse else (word + "$")
                fout.write(word + "\n")
                for i, (letter, elem) in enumerate(zip(adjusted_word, curr_attention)):
                    start, word_start = max(lm.history-i, 0), max(i - lm.history, 0)
                    fout.write("{}\t{}\n".format(
                        letter, " ".join("{}-{:.0f}".format(x[0], 100*x[1])
                                         for x in zip(adjusted_word[word_start:i], elem[start:]))))
                fout.write("\n")
    if params["test_file"] is not None:
        if "max_test_sents" in params:
            token_reading_params["max_sents"] = params["max_test_sents"]
        if symbols_has_features:
            test_data = read_tags_infile(params["test_file"], wrap=True,
                                         attach_tokens=symbols_has_tokens,
                                         **token_reading_params)
        else:
            test_data = read_input(params["test_file"], label_field="pos")
        predictions = lm.predict(test_data, return_letter_scores=True,
                                 return_log_probs=True, batch_size=BUCKET_SIZE)
        lengths = np.array([len(x[0]) for x in predictions])
        scores = np.array([x[1] for x in predictions])
        # print(lengths, scores)
        perplexity = np.sum(scores) / np.sum(lengths)
        # perplexity = np.mean(scores)
        print("{:4f}".format(perplexity))
        if params["outfile"] is not None:
            output_results(params["outfile"], test_data, predictions,
                           on_separate_lines=symbols_has_features)
        if params["detailed_outfile"] is not None:
            scores = lm.predict_proba(test_data)
            output_results(params["detailed_outfile"], test_data, predictions,
                           symbols=lm.vocabulary_.symbols_, probs=scores,
                           best_symbols_number=params.get("best_symbols_number", 1),
                           on_separate_lines=symbols_has_features)