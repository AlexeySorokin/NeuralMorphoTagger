"""
Contains implementation of cells and layers used in NeuralLM construction
"""
import numpy as np

import keras.backend as kb
import keras.layers as kl
from keras.engine import Layer, Model
from keras.engine.topology import InputSpec

from keras import initializers
from keras import regularizers
from keras import constraints

from .common import distributed_dot_softmax, distributed_transposed_dot
if kb.backend() == "theano":
    from .cells_theano import make_history_theano, make_context_theano
elif kb.backend() == "tensorflow":
    from .cells_tensorflow import batch_shifted_fill


def make_history(X, h, pad, flatten=False):
    if kb.backend() == "theano":
        answer = make_history_theano(X, h, pad, flatten=flatten)
    else:
        answer = batch_shifted_fill(X, h, pad, flatten=flatten)
    if not hasattr(answer, "_keras_shape") and hasattr(X, "_keras_shape"):
        if len(X._keras_shape) == 2:
            new_shape = X._keras_shape + (h,)
        elif not flatten:
            new_shape = X._keras_shape[:-1] + (h, X._keras_shape[-1])
        elif X._keras_shape[-1] is not None:
            new_shape = X._keras_shape[:-1] + (h * X._keras_shape[-1],)
        else:
            new_shape = X._keras_shape[:-1] + (None,)
        answer._keras_shape = new_shape
    return answer


def distributed_cell(inputs):
    """
    Creates a functional wrapper over RNN cell,
    applying it on each timestep without propagating hidden states over timesteps

    """
    assert len(inputs) == 2
    shapes = [elem._keras_shape for elem in inputs]
    # no shape validation, assuming all dims of inputs[0] and inputs[1] are equal
    input_dim, units, ndims = shapes[0][-1], shapes[1][-1], len(shapes[0])
    if ndims > 3:
        dims_order = (1,) + tuple(range(2, ndims)) + (2,)
        inputs = [kl.Permute(dims_order)(inputs[0]), kl.Permute(dims_order)(inputs[0])]
    first_shape, second_shape = shapes[0][2:], shapes[1][2:]
    cell = kl.GRUCell(units, input_shape=first_shape, implementation=0)
    if not cell.built:
        cell.build(first_shape)
    concatenated_inputs = kl.Concatenate()(inputs)
    def timestep_func(x):
        cell_inputs = x[...,:input_dim]
        cell_states = x[...,None,input_dim:]
        cell_output = cell.call(cell_inputs, cell_states)
        return cell_output[0]
    func = kl.TimeDistributed(kl.Lambda(timestep_func, output_shape=second_shape))
    answer = func(concatenated_inputs)
    if ndims > 3:
        reverse_dims_order = (1, ndims-1) + tuple(range(2,ndims-1))
        answer = kl.Permute(reverse_dims_order)(answer)
    return answer


class AttentionCell(Layer):
    """
    A layer collecting in each position a weighted sum of previous words embeddings
    where weights in the sum are calculated using attention
    """

    def __init__(self, left, input_dim, output_dim, right=0,
                 merge="concatenate", use_bias=False,
                 embeddings_initializer='uniform', embeddings_regularizer=None,
                 activity_regularizer=None, embeddings_constraint=None, **kwargs):
        if 'input_shape' not in kwargs:
            kwargs['input_shape'] = [(None, input_dim,), (None, output_dim)]
        super(AttentionCell, self).__init__(**kwargs)
        self.left = left
        self.output_dim = output_dim
        self.right = right
        self.merge = merge
        self.use_bias = use_bias
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.embeddings_regularizer = regularizers.get(embeddings_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.embeddings_constraint = constraints.get(embeddings_constraint)
        self.input_dim = input_dim
        self.input_spec = [InputSpec(shape=(None, input_dim)),
                           InputSpec(shape=(None, None, output_dim))]

    def build(self, input_shape):
        assert len(input_shape) == 2
        assert len(input_shape[0]) == 2 and len(input_shape[1])
        self.M = self.add_weight(shape=(self.input_dim, self.output_dim),
                                 initializer=self.embeddings_initializer,
                                 name='attention_embedding_1', dtype=self.dtype,
                                 regularizer=self.embeddings_regularizer,
                                 constraint=self.embeddings_constraint)
        self.C = self.add_weight(shape=(self.input_dim, self.output_dim),
                                 initializer=self.embeddings_initializer,
                                 name='attention_embedding_2', dtype=self.dtype,
                                 regularizer=self.embeddings_regularizer,
                                 constraint=self.embeddings_constraint)
        if self.use_bias:
            self.T = self.add_weight(shape=(self.left, self.output_dim),
                                     initializer=self.embeddings_initializer,
                                     name='bias', dtype=self.dtype,
                                     regularizer=self.embeddings_regularizer,
                                     constraint=self.embeddings_constraint)
        super(AttentionCell, self).build(input_shape)

    def call(self, inputs, **kwargs):
        assert isinstance(inputs, list) and len(inputs) == 2
        symbols, encodings = inputs[0], inputs[1]
        # contexts.shape = (M, T, left)
        contexts = make_history(symbols, self.left, symbols[:,:1])
        # M.shape = C.shape = (M, T, left, output_dim)
        M = kb.gather(self.M, contexts) # input embeddings
        C = kb.gather(self.C, contexts) # output embeddings
        if self.use_bias:
            M += self.T
        # p.shape = (M, T, input_dim)
        p = distributed_dot_softmax(M, encodings)
        # p._keras_shape = M._keras_shape[:2] + (self.)
        compressed_context = distributed_transposed_dot(C, p)
        if self.merge in ["concatenate", "sum"] :
            output_func = (kl.Concatenate() if self.merge == "concatenate"
                           else kl.Merge(mode='sum'))
            output = output_func([compressed_context, encodings])
        elif self.merge == "attention":
            output = compressed_context
        elif self.merge == "sigmoid":
            output = distributed_cell([compressed_context, encodings])
        return [output, p]

    def compute_output_shape(self, input_shape):
        first_shape, second_shape = input_shape
        if self.merge == "concatenate":
            output_shape = second_shape[:2] + (2*second_shape[2],)
        # elif self.merge in ["sum", "attention", "sigmoid"]:
        else:
            output_shape = second_shape
        p_shape = second_shape[:2] + (self.input_dim,)
        return [output_shape, p_shape]


class AttentionCell3D(Layer):
    """
    Attention cell applicable to 3D data
    """

    def __init__(self, left, input_dim, output_dim, right=0,
                 merge="concatenate", use_bias=False,
                 embeddings_initializer='uniform', embeddings_regularizer=None,
                 activity_regularizer=None, embeddings_constraint=None, **kwargs):
        if 'input_shape' not in kwargs:
            kwargs['input_shape'] = [(None, input_dim,), (None, output_dim)]
        super(AttentionCell3D, self).__init__(**kwargs)
        self.left = left
        self.output_dim = output_dim
        self.right = right
        self.merge = merge
        self.use_bias = use_bias
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.embeddings_regularizer = regularizers.get(embeddings_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.embeddings_constraint = constraints.get(embeddings_constraint)
        self.input_dim = input_dim
        self.input_spec = [InputSpec(shape=(None, None, input_dim)),
                           InputSpec(shape=(None, None, output_dim))]

    def build(self, input_shape):
        assert len(input_shape) == 2
        assert len(input_shape[0]) == 3 and len(input_shape[1]) == 3
        self.M = self.add_weight(shape=(self.input_dim, self.output_dim),
                                 initializer=self.embeddings_initializer,
                                 name='3Dattention_embedding_1', dtype=self.dtype,
                                 regularizer=self.embeddings_regularizer,
                                 constraint=self.embeddings_constraint)
        self.C = self.add_weight(shape=(self.input_dim, self.output_dim),
                                 initializer=self.embeddings_initializer,
                                 name='3Dattention_embedding_2', dtype=self.dtype,
                                 regularizer=self.embeddings_regularizer,
                                 constraint=self.embeddings_constraint)
        if self.use_bias:
            self.T = self.add_weight(shape=(self.left, self.output_dim),
                                     initializer=self.embeddings_initializer,
                                     name='bias', dtype=self.dtype,
                                     regularizer=self.embeddings_regularizer,
                                     constraint=self.embeddings_constraint)
        super(AttentionCell3D, self).build(input_shape)

    def call(self, inputs, **kwargs):
        assert isinstance(inputs, list) and len(inputs) == 2
        symbols, encodings = inputs[0], inputs[1]
        # contexts.shape = (M, T, left, input_dim)
        contexts = make_history(symbols, self.left, symbols[:,:1])
        # M.shape = C.shape = (M, T, left, output_dim)
        M = kb.dot(contexts, self.M) # input embeddings
        C = kb.dot(contexts, self.C) # output embeddings
        if self.use_bias:
            M += self.T
        p = distributed_dot_softmax(M, encodings)
        compressed_context = distributed_transposed_dot(C, p)
        if self.merge in ["concatenate", "sum"] :
            output_func = (kl.Concatenate() if self.merge == "concatenate"
                           else kl.Merge(mode='sum'))
            output = output_func([compressed_context, encodings])
        elif self.merge == "attention":
            output = compressed_context
        elif self.merge == "sigmoid":
            output = distributed_cell([compressed_context, encodings])
        return [output, p]

    def compute_output_shape(self, input_shape):
        first_shape, second_shape = input_shape
        if self.merge == "concatenate":
            output_shape = second_shape[:2] + (2*second_shape[2],)
        # elif self.merge in ["sum", "attention", "sigmoid"]:
        else:
            output_shape = second_shape
        p_shape = second_shape[:2] + (self.input_dim,)
        return [output_shape, p_shape]