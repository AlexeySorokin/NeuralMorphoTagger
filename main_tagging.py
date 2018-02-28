import sys
import json
import copy

import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from neural_LM.UD_preparation.extract_tags_from_UD import read_tags_infile, make_UD_pos_and_tag
from neural_tagging.neural_tagging import CharacterTagger, load_tagger

DEFAULT_NONE_PARAMS = ["model_file", "test_files", "outfiles", "train_files",
                       "dev_files", "dump_file", "save_file",
                       "prediction_files", "comparison_files"]
DEFAULT_PARAMS = {}
DEFAULT_DICT_PARAMS = ["model_params", "read_params", "predict_params", "vocabulary_files",
                       "train_read_params", "dev_read_params", "test_read_params"]


def read_config(infile):
    with open(infile, "r", encoding="utf8") as fin:
        from_json = json.load(fin)
    params = dict()
    for param in DEFAULT_NONE_PARAMS:
        params[param] = from_json.get(param)
    for param in DEFAULT_DICT_PARAMS:
        params[param] = from_json.get(param, dict())
    for param, default_value in DEFAULT_PARAMS.items():
        params[param] = from_json.get(param, default_value)
    for param, value in from_json.items():
        if param not in params:
            params[param] = value
    return params


def make_file_params_list(param, k, name="params"):
    if isinstance(param, str):
        param = [param]
    elif param is None:
        param = [None] * k
    if len(param) != k:
        Warning("You should pass the same number of {0} as test_files, "
                "setting {0} to None".format(name))
        param = [None] * k
    return param


def calculate_answer_probs(vocab, probs, labels):
    answer = [None] * len(labels)
    for i, (curr_probs, curr_labels) in enumerate(zip(probs, labels)):
        m = len(curr_labels)
        curr_label_indexes = [vocab.toidx(label) for label in curr_labels]
        answer[i] = curr_probs[np.arange(m), curr_label_indexes]
    return answer


def output_predictions(outfile, data, labels):
    with open(outfile, "w", encoding="utf8") as fout:
        for sent, sent_labels in zip(data, labels):
            for j, (word, label) in enumerate(zip(sent, sent_labels), 1):
                label, tag = make_UD_pos_and_tag(label)
                fout.write("{}\t{}\t{}\t{}\n".format(j, word, label, tag))
            fout.write("\n")


def output_results(outfile, data, pred_labels, corr_labels, probs, corr_probs):
    with open(outfile, "w", encoding="utf8") as fout:
        for i, (sent, sent_pred_labels, sent_labels, sent_probs, sent_corr_probs)\
                in enumerate(zip(data, pred_labels, corr_labels, probs, corr_probs)):
            is_correct = (sent_pred_labels == sent_labels)
            total_prob = -np.sum(np.log(sent_probs))
            total_corr_prob = -np.sum(np.log(sent_corr_probs))
            fout.write("{:.3f}\t{:.3f}\n".format(total_prob, total_corr_prob))
            if not is_correct:
                fout.write("INCORRECT\n")
            for j, (word, pred_tag, corr_tag, pred_prob, corr_prob) in\
                    enumerate(zip(sent, sent_pred_labels,
                                  sent_labels, sent_probs, sent_corr_probs)):
                curr_format_string =\
                    "{0}\t{1}\t{2}" + ("\tERROR\n" if pred_tag != corr_tag else "\n")
                fout.write(curr_format_string.format("".join(word), corr_tag, pred_tag))
            fout.write("\n")


def make_output(cls, test_data, test_labels, predictions, probs,
                outfile=None, comparison_file=None):
    corr, total, corr_sent = 0, 0, 0
    for pred, test in zip(predictions, test_labels):
        total += len(test)
        curr_corr = sum(int(x == y) for x, y in zip(pred, test))
        corr += curr_corr
        corr_sent += int(len(test) == curr_corr)
    print("Точность {:.2f}: {} из {} меток".format(100 * corr / total, corr, total))
    print("Точность по предложениям {:.2f}: {} из {} предложений".format(
        100 * corr_sent / len(test_labels), corr_sent, len(test_labels)))
    if outfile is not None:
        with open(outfile, "w", encoding="utf8") as fout:
            for sent, pred, test in zip(test_data, predictions, test_labels):
                for word, pred_tag, corr_tag in zip(sent, pred, test):
                    format_string = "{0}\t{1}\t{2}" + ("\tERROR\n" if pred_tag != corr_tag else "\n")
                    fout.write(format_string.format("".join(word), corr_tag, pred_tag))
                fout.write("\n")
    if comparison_file is not None:
        prediction_probs = calculate_answer_probs(cls.tags_, probs, predictions)
        corr_probs = calculate_answer_probs(cls.tags_, probs, test_labels)
        output_results(comparison_file, test_data, predictions, test_labels,
                       prediction_probs, corr_probs)


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
    params["predict_params"]["return_probs"] = True
    if params["train_files"] is not None:
        cls = CharacterTagger(**params["model_params"])
        train_read_params = copy.deepcopy(params["read_params"])
        train_read_params.update(params["train_read_params"])
        train_data = []
        for train_file in params["train_files"]:
            train_data += read_tags_infile(train_file, read_words=True, **train_read_params)
        train_data, train_labels = [x[0] for x in train_data], [x[1] for x in train_data]
        if params["dev_files"] is not None:
            dev_read_params = copy.deepcopy(params["read_params"])
            dev_read_params.update(params["dev_read_params"])
            dev_data = []
            for dev_file in params["dev_files"]:
                dev_data += read_tags_infile(dev_file, read_words=True, **dev_read_params)
            dev_data, dev_labels = [x[0] for x in dev_data], [x[1] for x in dev_data]
        else:
            dev_data, dev_labels = None, None
        cls.train(train_data, train_labels, dev_data, dev_labels,
                  model_file=params["model_file"], save_file=params["save_file"],
                  lm_file=params["lm_file"], **params["vocabulary_files"])
    elif params["load_file"] is not None:
        cls, train_data = load_tagger(params["load_file"]), None
    else:
        raise ValueError("Either train_file or load_file should be given")
    if params["save_file"] is not None and params["dump_file"] is not None:
        cls.to_json(params["save_file"], params["dump_file"])
    if params["test_files"] is not None:
        test_read_params = copy.deepcopy(params["read_params"])
        test_read_params.update(params["test_read_params"])
        # defining output files
        test_files = params["test_files"]
        if isinstance(test_files, str):
            test_files = [test_files]
        prediction_files = make_file_params_list(params["prediction_files"], len(test_files),
                                                 name="prediction_files")
        outfiles = make_file_params_list(params["outfiles"], len(test_files),
                                                 name="outfiles")
        comparison_files = make_file_params_list(params["comparison_files"], len(test_files),
                                                 name="comparison_files")
        # loading language model if available
        for (test_file, prediction_file, outfile, comparison_file) in zip(
                test_files, prediction_files, outfiles, comparison_files):
            test_data, source_data = read_tags_infile(
                test_file, read_words=True, return_source_words=True, **test_read_params)
            if not test_read_params.get("read_only_words", False):
                test_data, test_labels = [x[0] for x in test_data], [x[1] for x in test_data]
            else:
                test_labels = None
            cls_predictions = cls.predict(test_data, **params["predict_params"])
            predictions, probs = cls_predictions[:2]
            basic_probs = cls_predictions[2] if len(cls_predictions) > 2 else None
            if prediction_file is not None:
                output_predictions(prediction_file, source_data, predictions)
            if test_labels is not None:
                make_output(cls, test_data, test_labels, predictions,
                            probs, outfile, comparison_file)

