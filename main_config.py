import sys
import getopt
import json

import numpy as np
from keras.callbacks import EarlyStopping

from neural_LM.neural_lm import NeuralLM, read_input, load_lm
from neural_LM.UD_preparation.read_tags import read_tags_input

SHORT_OPTS = "HLFm:t:v:p:o:s:d:l:h:aeO:"
LONG_OPTS = ["batch_size=", "nepochs=", "validation=",
             "embeddings=", "memory-embeddings=", "rnn-size="]
BUCKET_SIZE = 256


DEFAULT_NONE_PARAMS = ["model_file", "test_file", "outfile", "attention_file",
                       "train_files", "dev_files", "dump_file", "save_file"]
DEFAULT_PARAMS = {"symbols_has_features": False}




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
    return params



if __name__ == '__main__':
    if len(sys.argv[1:]) != 1:
        sys.exit("Usage: main.py <config json file>")
    params = read_config(sys.argv[1])
    if "stop_callback" in params:
        stop_callback = EarlyStopping(**params["stop_callback"])
        callbacks = [stop_callback]
    else:
        callbacks = []
    symbols_has_features = params["symbols_has_features"]
    if params["train_files"] is not None:
        # maximal history of 5 symbols is effective
        lm = NeuralLM(**params["model_params"])
        train_data = []
        for train_file in params["train_files"]:
            if symbols_has_features:
                train_data += read_tags_input(train_file)
            else:
                train_data += read_input(train_file, label_field="pos")
        if params["dev_files"] is not None:
            dev_data = []
            for dev_file in params["dev_files"]:
                if symbols_has_features:
                    dev_data += read_tags_input(dev_file)
                else:
                    dev_data += read_input(dev_file, label_field="pos")
        else:
            dev_data = None
        lm.train(train_data, X_dev=dev_data, model_file=params["model_file"])
    elif params["load_file"] is not None:
        lm, train_data = load_lm(params["load_file"]), None
    else:
        raise ValueError("Either train_file or load_file should be given")
    dump_file = params.get("dump_file", params["model_file"])
    if params["save_file"] is not None and dump_file is not None:
        lm.to_json(params["save_file"], dump_file)
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
        if symbols_has_features:
            test_data = read_tags_input(params["test_file"])
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
            with open(params["outfile"], "w", encoding="utf8") as fout:
                for (word, tag, feats), (symbol_scores, score) in zip(test_data, predictions):
                    if lm.reverse:
                        # слова читались с конца
                        symbol_scores = symbol_scores[::-1]
                    adjusted_word = ("^" + word) if lm.reverse else (word + "$")
                    fout.write("{}\t{}\t{:.2f}".format(
                        word, " ".join("{}-{:.3f}".format(x, y)
                                       for x, y in zip(adjusted_word + "$", symbol_scores)),
                        score))
