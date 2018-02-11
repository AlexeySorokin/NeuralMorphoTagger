"""
File containing common operation with theano structures
required by NeuralLM, but possibly useful for other modules
"""

import numpy as np

import keras.backend as kb


AUXILIARY = ['PAD', 'BEGIN', 'END', 'UNKNOWN']
PAD, BEGIN, END, UNKNOWN = 0, 1, 2, 3

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

def shifted_repeat(x, k):
    def _scan_shifted_repeat(a, i, b):
        m, n = b.shape[1:3]
        start = n * i
        a_new = tT.set_subtensor(a[:,i:, start:start+n], b[:,:m-i])
        return [a_new, i+1]
    x_new = kb.zeros_like(kb.repeat_elements(x, k, axis=2))
    results, updates = theano.scan(
        _scan_shifted_repeat, non_sequences=[x],
        outputs_info=[x_new, 0], n_steps=tT.smallest(k, x.shape[1]))
    return results[0][-1]

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


# def test_transposed_dot():
#     A = np.reshape(np.arange(24, dtype=float), (2, 4, 3))
#     H = np.array(([1,-2, 1,0], [1, -3, 2, 1]), dtype=float)
#     a = tT.dtensor3()
#     h = tT.dmatrix()
#     b = distributed_transposed_dot(a, h)
#     f = theano.function([a, h], [b], on_unused_input='warn')
#     print(f(A, H))

