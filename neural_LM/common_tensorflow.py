import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn


def sequence_length(seq, end_value):
    """
    Returns the length of the sequence as the position of END token plus 1
    """
    return tf.argmax(tf.cast(tf.equal(seq, end_value), tf.int32), axis=1) + 1

def shifted_fill(a, k, pad):
    a = tf.convert_to_tensor(a)
    pad = tf.convert_to_tensor(pad)
    # k = tf.convert_to_tensor(k)
    padding = tf.tile([pad], [k-1] + [1] * (pad.get_shape().ndims))
    a_padded = tf.concat([padding, a], axis=0)
    answer_shape = tf.concat([[0, k], tf.cast(a.shape[1:], tf.int32)], axis=0)
    i, answer = k-1, tf.zeros(answer_shape, dtype=a.dtype)
    cond = lambda i, ans, a, k: i < a_padded.shape[0]
    body = lambda i, ans, a, k: (
        i+1, tf.concat([ans, tf.expand_dims(a_padded[i-k+1:i+1], 0)], axis=0), a, k)
    ans_shape = tf.TensorShape([None, k]).concatenate(a_padded.shape[1:])
    _, answer_, _, _ = tf.while_loop(cond, body, [i, answer, a_padded, k],
                                     shape_invariants=[tf.TensorShape([]), ans_shape,
                                                       a_padded.shape, tf.TensorShape([])])
    return answer_