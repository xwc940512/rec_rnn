# Temporarily copied most of the code from
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn_cell.py
#
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import six

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh

from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.rnn_cell import RNNCell


class RecLSTMCell(RNNCell):
    def __init__(self, num_units, forget_bias=1.0, input_size=None):
        self._num_units = num_units
        self._input_size = num_units if input_size is None else input_size
        self._forget_bias = forget_bias

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return 2 * self._num_units

    def __call__(self, inputs, state, scope=None):
        with vs.variable_scope(scope or type(self).__name__):
            # Parameters of gates are concatenated into one multiply for efficiency.
            input_i, input_u = array_ops.split(1, 2, inputs)
            c, h = array_ops.split(1, 2, state)
            concat = _linear([input_i, h], 4 * self._num_units, True)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = array_ops.split(1, 4, concat)

            mat_uf = vs.get_variable("mat_uf", [self.input_size, self.output_size])
            user_f = math_ops.matmul(input_u, mat_uf)

            mat_ui = vs.get_variable("mat_ui", [self.input_size, self.output_size])
            user_i = math_ops.matmul(input_u, mat_ui)


            new_c = c * sigmoid(f + self._forget_bias + user_f) + sigmoid(i + user_i) * tanh(j)
            new_h = tanh(new_c) * sigmoid(o)
            return array_ops.concat(1, [new_h, input_u]), array_ops.concat(1, [new_c, new_h])


def _linear(args, output_size, bias, bias_start=0.0, scope=None):
    if args is None or (_is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not _is_sequence(args):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
        if not shape[1]:
            raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
            total_arg_size += shape[1]

    # Now the computation.
    with vs.variable_scope(scope or "Linear"):
        matrix = vs.get_variable("Matrix", [total_arg_size, output_size])
        if len(args) == 1:
            res = math_ops.matmul(args[0], matrix)
        else:
            res = math_ops.matmul(array_ops.concat(1, args), matrix)
        if not bias:
            return res
        bias_term = vs.get_variable(
            "Bias", [output_size],
            initializer=init_ops.constant_initializer(bias_start))
    return res + bias_term


def _is_sequence(seq):
    return isinstance(seq, collections.Sequence) and not isinstance(seq, six.string_types)