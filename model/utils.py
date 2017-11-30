import tensorflow as tf
import os
from sklearn.preprocessing import StandardScaler
import numpy as np

from tensorflow.python.ops import init_ops
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import moving_averages


def fill_na(x, fillval=0):
    fill = tf.ones_like(x) * fillval
    return tf.where(tf.is_finite(x), x, fill)


def nanmean(x, axis=None):
    x_filled = fill_na(x, 0)
    x_sum = tf.reduce_sum(x_filled, axis=axis)
    x_count = tf.reduce_sum(tf.cast(tf.is_finite(x), tf.float32), axis=axis)
    return tf.div(x_sum, x_count)


def _prepend_edge(tensor, pad_amt, axis=1):
    '''
    This function is intented to add 'reflective' padding to a 4d Tensor across
        the height and width dimensions

    Parameters
    ----------
    tensor: Tensor with rank 4
    pad_amt: Integer
    axis: Integer
        Must be in (1,2)
    '''
    if axis not in (1, 2):
        raise ValueError("Axis must equal 0 or 1. Axis is set to %i" % axis)

    if axis == 1:
        concat_dim = 2
    else:
        concat_dim = 1

    begin = [0, 0, 0, 0]
    end = [-1, -1, -1, -1]
    end[axis] = 1

    edges = pad_amt * [tf.slice(tensor, begin, end)]
    # edges = pad_amt * [tf.slice(tensor, begin, end) * 0]
    if len(edges) > 1:
        padding = tf.concat(axis=axis, values=edges)
    else:
        padding = edges[0]

    tensor_padded = tf.concat(axis=axis, values=[padding, tensor])
    return tensor_padded


def _append_edge(tensor, pad_amt, axis=1):
    '''
    This function is intented to add 'reflective' padding to a 4d Tensor across
        the height and width dimensions

    Parameters
    ----------
    tensor: Tensor with rank 4
    pad_amt: Integer
    axis: Integer
        Must be in (1,2)
    '''
    if axis not in (1, 2):
        raise ValueError("Axis must equal 0 or 1. Axis is set to %i" % axis)

    if axis == 1:
        concat_dim = 2
    else:
        concat_dim = 1

    begin = [0, 0, 0, 0]
    end = [-1, -1, -1, -1]
    begin[axis] = tf.shape(tensor)[axis] - 1  # go to the end

    edges = pad_amt * [tf.slice(tensor, begin, end)]
    # edges = pad_amt * [tf.slice(tensor, begin, end) * 0]
    if len(edges) > 1:
        padding = tf.concat(axis=axis, values=edges)
    else:
        padding = edges[0]

    tensor_padded = tf.concat(axis=axis, values=[tensor, padding])
    return tensor_padded


def replicate_padding(tensor, pad_amt):
    if isinstance(pad_amt, int):
        pad_amt = [pad_amt] * 2
    for axis, p in enumerate(pad_amt):
        tensor = _prepend_edge(tensor, p, axis=axis + 1)
        tensor = _append_edge(tensor, p, axis=axis + 1)
    return tensor
