from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import array_ops
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh

from rec_rnn.util.util import linear

from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.rnn_cell import RNNCell


class RecRNNCell(object):
    def __call__(self, inputs_i, inputs_u, state, scope=None):
        raise NotImplementedError("Abstract method")

    @property
    def input_size(self):
        raise NotImplementedError("Abstract method")

    @property
    def output_size(self):
        raise NotImplementedError("Abstract method")

    @property
    def state_size(self):
        raise NotImplementedError("Abstract method")

    def zero_state(self, batch_size, dtype):
        zeros = array_ops.zeros(
            array_ops.pack([batch_size, self.state_size]), dtype=dtype)
        zeros.set_shape([None, self.state_size])
        return zeros


class RecLSTMCell(RecRNNCell):
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

    def __call__(self, inputs_i, inputs_u, state, scope=None):
        with vs.variable_scope(scope or type(self).__name__):
            # Parameters of gates are concatenated into one multiply for efficiency.
            c, h = array_ops.split(1, 2, state)
            concat = linear([inputs_i, inputs_u, h], 4 * self._num_units, True)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = array_ops.split(1, 4, concat)

            new_c = c * sigmoid(f + self._forget_bias) + sigmoid(i) * tanh(j)
            new_h = tanh(new_c) * sigmoid(o)
            return new_h, array_ops.concat(1, [new_c, new_h])


class MultiRecRNNCell(RecRNNCell):
  def __init__(self, cells):
    if not cells:
      raise ValueError("Must specify at least one cell for MultiRNNCell.")
    for i in xrange(len(cells) - 1):
      if cells[i + 1].input_size != cells[i].output_size:
        raise ValueError("In MultiRNNCell, the input size of each next"
                         " cell must match the output size of the previous one."
                         " Mismatched output size in cell %d." % i)
    self._cells = cells

  @property
  def input_size(self):
    return self._cells[0].input_size

  @property
  def output_size(self):
    return self._cells[-1].output_size

  @property
  def state_size(self):
    return sum([cell.state_size for cell in self._cells])

  def __call__(self, inputs_i, inputs_u, state, scope=None):
    """Run this multi-layer cell on inputs, starting from state."""
    with vs.variable_scope(scope or type(self).__name__):  # "MultiRNNCell"
      cur_state_pos = 0
      cur_inp = inputs_i
      new_states = []
      for i, cell in enumerate(self._cells):
        with vs.variable_scope("Cell%d" % i):
          cur_state = array_ops.slice(
              state, [0, cur_state_pos], [-1, cell.state_size])
          cur_state_pos += cell.state_size
          cur_inp, new_state = cell(cur_inp, inputs_u, cur_state)
          new_states.append(new_state)
    return cur_inp, array_ops.concat(1, new_states)


