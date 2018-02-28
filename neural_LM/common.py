"""
File containing common operation with theano structures
required by NeuralLM, but possibly useful for other modules
"""

import numpy as np
import itertools

import keras.backend as kb
from keras.callbacks import Callback

EPS = 1e-15

AUXILIARY = ['PAD', 'BEGIN', 'END', 'UNKNOWN']
AUXILIARY_CODES = PAD, BEGIN, END, UNKNOWN = 0, 1, 2, 3


def to_one_hot(x, k):
    """
    Takes an array of integers and transforms it
    to an array of one-hot encoded vectors
    """
    unit = np.eye(k, dtype=int)
    return unit[x]

def repeat_(x, k):
    tile_factor = [1, k] + [1] * (kb.ndim(x) - 1)
    return kb.tile(x[:,None,:], tile_factor)

def distributed_transposed_dot(C, P):
    """
    Calculates for each timestep the weighted sum
    of the embeddings C_ according to the probability distribution P_

    C: a tensor of embeddings of shape
        batch_size * timesteps * history * embeddings_shape or
        timesteps * history * embeddings_shape
    P: a tensor of attention probabilities of shape
        batch_size * timesteps * history or
        timesteps * history

    Returns:
    ---------------
    answer: a tensor of weighted embeddings of shape
        batch_size * timesteps * embeddings_shape or
        timesteps * embeddings_shape
    """
    p_dims_number = int(kb.ndim(P))
    C_shape = tuple((kb.shape(C)[i] for i in range(kb.ndim(C))))
    # new_P_shape = (-1,) + tuple(kb.shape(P)[2:])
    C_ = kb.reshape(C, (-1,) + C_shape[p_dims_number-1:])
    P_ = kb.reshape(P, (-1, kb.shape(P)[p_dims_number-1]))
    answer_shape = C_shape[:p_dims_number-1] + C_shape[p_dims_number:]
    answer = kb.reshape(kb.batch_dot(C_, P_, axes=[1, 1]), answer_shape)
    if not hasattr(answer, "_keras_shape") and hasattr(C, "_keras_shape"):
        answer._keras_shape = C._keras_shape[:-2] + (C._keras_shape[-1],)
    return answer

def distributed_dot_softmax(M, H):
    """
    Obtains a matrix m of recent embeddings
    and the hidden state h of the lstm
    and calculates the attention distribution over embeddings
    p = softmax(<m, h>)

    M: embeddings tensor of shape
        (batch_size, timesteps, history_length, units) or
        (timesteps, history_length, units)
    H: hidden state tensor of shape
        (batch_size, timesteps, units) or
        (timesteps, units)
    :return:
    """
    # flattening all dimensions of M except the last two ones
    M_shape = kb.print_tensor(kb.shape(M))
    M_shape = tuple((M_shape[i] for i in range(kb.ndim(M))))
    new_M_shape = (-1,) + M_shape[-2:]
    H_shape = kb.print_tensor(kb.shape(H))
    new_H_shape = (-1, H_shape[-1])
    M_ = kb.reshape(M, new_M_shape)
    # new_H_shape = kb.concatenate([np.array([-1]), kb.shape(H)[-2:]], axis=0)
    H_ = kb.reshape(H, new_H_shape)
    energies = kb.batch_dot(M_, H_, axes=[2, 1])
    # Tensor representing shape is not iterable with tensorflow backend
    answer = kb.reshape(kb.softmax(energies), M_shape[:-1])
    if not hasattr(answer, "_keras_shape") and hasattr(M, "_keras_shape"):
        answer._keras_shape = M._keras_shape[:-1]
    return answer


class CustomCallback(Callback):

    def __init__(self):
        super(CustomCallback, self).__init__()

    def on_train_begin(self, logs=None):
        self.verbose = self.params['verbose']
        self.epochs = self.params['epochs']
        self.train_losses = []
        self.val_losses = []
        self.best_loss = np.inf

    def on_epoch_begin(self, epoch, logs=None):
        print('Epoch %d/%d' % (epoch + 1, self.epochs))

    def on_epoch_end(self, epoch, logs=None):
        self.train_losses.append(logs["loss"])
        self.val_losses.append(logs["val_loss"])
        print("loss: {:.4f}, val_loss: {:.4f}".format(self.train_losses[-1], self.val_losses[-1]))
        if self.val_losses[-1] < self.best_loss:
            self.best_loss = self.val_losses[-1]
            print(", best loss\n")
        else:
            print("\n")


def generate_data(X, indexes_by_buckets, output_symbols_number,
                  batch_size=None, use_last=True, has_answer=True,
                  shift_answer=False, shuffle=True, yield_weights=True,
                  duplicate_answer=False):
    fields_number = len(X[0]) - int(has_answer and not use_last)
    answer_index = 0 if use_last else -1 if has_answer else None
    if batch_size is None:
        batches_indexes = [(i, 0) for i in range(len(indexes_by_buckets))]
    else:
        batches_indexes = list(itertools.chain.from_iterable(
            (((i, j) for j in range(0, len(bucket), batch_size))
             for i, bucket in enumerate(indexes_by_buckets))))
    total_arrays_size = sum(np.count_nonzero(X[j][answer_index] != PAD) - 1
                            for elem in indexes_by_buckets for j in elem)
    total_data_length = sum(len(elem) for elem in indexes_by_buckets)
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
            if has_answer:
                indexes_to_yield = np.array([X[j][answer_index] for j in bucket_indexes])
                if shift_answer:
                    padding = np.full(shape=(end - start, 1), fill_value=PAD)
                    indexes_to_yield = np.hstack((indexes_to_yield[:,1:], padding))
                y_to_yield = to_one_hot(indexes_to_yield, output_symbols_number)
                weights_to_yield = np.ones(shape=(end - start,), dtype=np.float32)
                if yield_weights:
                    weights_to_yield *= total_data_length * indexes_to_yield.shape[1]
                    weights_to_yield /= total_arrays_size
                if duplicate_answer:
                    y_to_yield = [y_to_yield, y_to_yield]
                    weights_to_yield = [weights_to_yield, weights_to_yield]
                yield (to_yield, y_to_yield, weights_to_yield)
            else:
                yield to_yield

