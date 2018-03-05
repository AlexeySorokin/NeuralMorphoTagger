# NeuralMorphoTagger
The implementation of neural morphological tagger architectures using Keras with TensorFlow backend.

Implements the model from Georg Heigold, Guenter Neumann, Josef van Genabith An extensive empirical evaluation of character-based morphological tagging for 14 languages <http://www.aclweb.org/anthology/E17-1048>

Basic usage: python3 main_tagging.py \<config file\>\
Config example: neural_tagging/config/config_ud_hungarian.json
Data format: Universal Dependencies, <http://universaldependencies.org>, see the example files in data directory

Basic config fields:
* train_files: the list of Universal Dependencies files to train. They must be tokenized already.
* dev_files: the list of Universal Dependencies files to validate. In case there is no separate validation file, provide nonzero validation_split in __model_params__
* test_files: the list of test files. They could be either in UD format or in one-word-per-line format. In the latter case add the dictionary field __test_read_params__ containing read_only_words=true and word_column=0.
* prediction_files: the list of result files. Each line of result file contains four fields: word index in sentence, the word itself, its POS tag, morphological fields tag.
* model_file: the .hdf5 file to save model weights
* save_file: the json file to save model configuration
* load_file: the json file to load model configuration
* model_params: parameters of the training model, see the example config

  
