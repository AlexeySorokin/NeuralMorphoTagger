"""
File containing common operation with keras.backend objects
"""

import numpy as np
import itertools

import keras.backend as kb

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

