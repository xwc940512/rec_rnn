from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from rec_rnn.rec_lstm_cell import RecLSTMCell


class RecRNN(object):
    def __init__(self, is_training, config):
        self._is_training = is_training
        self._config = config

        self._input_i = tf.placeholder(tf.int32, [config.batch_size, config.num_steps])
        self._input_u = tf.placeholder(tf.int32, [config.batch_size, config.num_steps])
        self._targets = tf.placeholder(tf.int32, [config.batch_size, config.num_steps])

        self._lstm_cell = lstm_cell = self.define_lstm_cell()

        self._initial_state = lstm_cell.zero_state(config.batch_size, tf.float32)

        #self._embedded_i, self._embedded_u = self.define_input()
        self._embedding = self.define_input()
        self._outputs, self._usi, self._final_state = self.define_output(self._initial_state)

        self._cost = self.define_cost()

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        self._train_op = self.define_training()

    def define_input(self):
        with tf.device("/cpu:0"):
            embedding_i = tf.get_variable("embedding_i", [self.config.item_dim, self.config.hidden_size])
            embedded_i = tf.nn.embedding_lookup(embedding_i, self.input_i)

            embedding_u = tf.get_variable("embedding_u", [self.config.user_dim, self.config.hidden_size])
            embedded_u = tf.nn.embedding_lookup(embedding_u, self.input_u)

        if self.is_training and self.config.keep_prob < 1.0:
            embedded_i = tf.nn.dropout(embedded_i, self.config.keep_prob)
            embedded_u = tf.nn.dropout(embedded_u, self.config.keep_prob)

        return tf.concat(2, [embedded_i, embedded_u])

    def define_output(self, state):
        outputs = []
        usi = []
        with tf.variable_scope("RNN"):
            for time_step in range(self.config.num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = self.lstm_cell(self.embedding[:, time_step, :], state)
                h, u = tf.split(1, 2, cell_output)
                outputs.append(h)
                usi.append(u)

        return [tf.reshape(tf.concat(1, outputs), [-1, self.config.hidden_size]),
                tf.reshape(tf.concat(1, usi), [-1, self.config.hidden_size]),
                state]

    def define_lstm_cell(self):
        #num_units = self.config.hidden_size * 2
        lstm_cell = RecLSTMCell(self.config.hidden_size, forget_bias=0.0)
        if self.is_training and self.config.keep_prob < 1.0:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.config.keep_prob)

        return tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * self.config.num_layers)


    def define_cost(self):
        L_0 = tf.get_variable("l_0", [self.config.hidden_size, self.config.item_dim])
        b = tf.get_variable("b", [self.config.item_dim])

        self._logits = logits = tf.matmul(self.outputs + self._usi, L_0) + b
        loss = tf.nn.seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(self.targets, [-1])],
            [tf.ones([self.config.batch_size * self.config.num_steps])]
        )

        return tf.reduce_sum(loss) / self.config.batch_size

    def define_training(self):
        trainable_vars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, trainable_vars), self.config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self.lr)

        return optimizer.apply_gradients(zip(grads, trainable_vars))

    def assign_lr(self, session, lr):
        session.run(tf.assign(self.lr, lr))

    @property
    def lstm_cell(self):
        return self._lstm_cell

    @property
    def embedding(self):
        return self._embedding

    @property
    def embedded_i(self):
        return self._embedded_i

    @property
    def embedded_u(self):
        return self._embedded_u

    @property
    def outputs(self):
        return self._outputs

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def final_state(self):
        return self._final_state

    @property
    def cost(self):
        return self._cost

    @property
    def train_op(self):
        return self._train_op

    @property
    def lr(self):
        return self._lr

    @property
    def is_training(self):
        return self._is_training

    @property
    def input_i(self):
        return self._input_i

    @property
    def input_u(self):
        return self._input_u

    @property
    def targets(self):
        return self._targets

    @property
    def config(self):
        return self._config

