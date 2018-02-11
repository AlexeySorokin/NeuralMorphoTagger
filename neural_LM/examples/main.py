import sys
import getopt

import numpy as np
from keras.callbacks import EarlyStopping

from neural_LM.neural_lm import NeuralLM, read_input, load_lm
from neural_LM.UD_preparation.read_tags import read_tags_input

SHORT_OPTS = "HLFm:t:v:p:o:s:d:l:h:aeO:"
BUCKET_SIZE = 256


if __name__ == '__main__':
    opts, args = getopt.getopt(sys.argv[1:], SHORT_OPTS)
    if len(args) > 0:
        print("Usage: keras_lm.py [-s save_file] [-p test_file] [-t train_file]")
    model_file, test_file, outfile, train_files, dev_files = None, None, None, None, None
    save_file, dump_file, load_file, attention_outfile = None, None, None, None
    history, use_attention, use_embeddings = 3, False, False
    symbols_has_features, use_label, use_feats = False, True, True
    for opt, val in opts:
        if opt == "-H":
            symbols_has_features = True
        elif opt == "-L":
            use_label, use_feats = False, False
        elif opt == "-F":
            use_feats = False
        elif opt == "-m":
            model_file = val
        elif opt == "-t":
            train_files = val.split(",")
        elif opt == "-v":
            dev_files = val.split(",")
        elif opt == "-p":
            test_file = val
        elif opt == "-o":
            outfile = val
        elif opt == "-s":
            save_file = val
        elif opt == "-d":
            dump_file = val
        elif opt == "-l":
            load_file = val
        elif opt == "-h":
            history = int(val)
        elif opt == "-a":
            use_attention = True
        elif opt == "-e":
            use_embeddings = True
        elif opt == "-O":
            attention_outfile = val
    stop_callback = EarlyStopping(monitor='val_loss', patience=25, verbose=1)
    callbacks = [stop_callback]
    if train_files is not None:
        # maximal history of 5 symbols is effective
        lm = NeuralLM(symbols_has_features=symbols_has_features, use_label=use_label, use_feats=use_feats,
                      history=history, use_attention=use_attention,
                      attention_activation="concatenate", memory_embeddings_size=64,
                      use_embeddings=use_embeddings, embeddings_size=64,
                      use_output_rnn=True, use_attention_bias=True,
                      nepochs=100, batch_size=61, rnn_size=64,
                      verbose=1, callbacks=callbacks, random_state=197)
        train_data = []
        for train_file in train_files:
            if symbols_has_features:
                train_data += read_tags_input(train_file)
            else:
                train_data += read_input(train_file, label_field="pos")
        if dev_files is not None:
            dev_data = []
            for dev_file in dev_files:
                if symbols_has_features:
                    dev_data += read_tags_input(dev_file)
                else:
                    dev_data += read_input(dev_file, label_field="pos")
        else:
            dev_data = None
        lm.train(train_data, X_dev=dev_data, model_file=model_file)
    elif load_file is not None:
        lm, train_data = load_lm(load_file), None
    else:
        raise ValueError("Either train_file or load_file should be given")
    if dump_file is None:
        dump_file = model_file
    if save_file is not None and dump_file is not None:
        lm.to_json(save_file, dump_file)
    if attention_outfile is not None:
        if train_data is None:
            print("Impossible to output attention: no training data")
        attention_scores = lm.predict_attention(train_data, batch_size=128)
        with open(attention_outfile, "w", encoding="utf8") as fout:
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
    if test_file is not None:
        if symbols_has_features:
            test_data = read_tags_input(test_file)
        else:
            test_data = read_input(test_file, label_field="pos")
        predictions = lm.predict(test_data, return_letter_scores=True,
                                 return_log_probs=True, batch_size=BUCKET_SIZE)
        lengths = np.array([len(x[0]) for x in predictions])
        scores = np.array([x[1] for x in predictions])
        # print(lengths, scores)
        perplexity = np.sum(scores) / np.sum(lengths)
        # perplexity = np.mean(scores)
        print("{:4f}".format(perplexity))
        if outfile is not None:
            with open(outfile, "w", encoding="utf8") as fout:
                for (word, tag, feats), (symbol_scores, score) in zip(test_data, predictions):
                    if lm.reverse:
                        # слова читались с конца
                        symbol_scores = symbol_scores[::-1]
                    adjusted_word = ("^" + word) if lm.reverse else (word + "$")
                    fout.write("{}\t{}\t{:.2f}".format(
                        word, " ".join("{}-{:.3f}".format(x, y)
                                       for x, y in zip(adjusted_word + "$", symbol_scores)),
                        score))
