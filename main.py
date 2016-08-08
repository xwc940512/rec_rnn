from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import unittest

import numpy as np
import tensorflow as tf
from data_sets.io.movielens.ml_reader import MlReader

from rec_rnn.rec_rnn import RecRNN


def test_network():
    reader, data_path = get_setup()

    raw_data = reader.raw_data(data_path)
    train_data, valid_data, test_data, item_dim, user_dim = raw_data

    print(item_dim)
    print(user_dim)

    config = Config()
    config.item_dim = item_dim
    config.user_dim = user_dim
    eval_config = Config()
    eval_config.batch_size = 1
    eval_config.num_steps = 1
    eval_config.item_dim = item_dim
    eval_config.user_dim = user_dim

    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

        with tf.variable_scope("model", reuse=None, initializer=initializer):
            model_training = RecRNN(is_training=True, config=config)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            model_valid = RecRNN(is_training=False, config=config)
            model_test = RecRNN(is_training=False, config=eval_config)

        tf.initialize_all_variables().run()

        for i in range(config.max_max_epoch):
            training_epoch(model_training, session, train_data, config, reader, i)
            validation_epoch(model_valid, session, valid_data, reader, i)
        validation_epoch(model_test, session, test_data, reader)


def training_epoch(model, session, data, config, reader, epoch):
    lr_decay = config.lr_decay ** max(epoch - config.max_epoch, 0.0)
    model.assign_lr(session, config.learning_rate * lr_decay)

    print("Epoch: %d Learning rate: %.3f" % (epoch + 1, session.run(model.lr)))
    perplexity = run_epoch(session, model, data, model.train_op, reader, verbose=True)
    print("Epoch: %d Train Perplexity: %.3f" % (epoch + 1, perplexity))


def validation_epoch(model, session, data, reader, epoch=None):
    perplexity = run_epoch(session, model, data, tf.no_op(), reader)
    if epoch is not None:
        print("Epoch: %d Valid Perplexity: %.3f" % (epoch + 1, perplexity))
    else:
        print("Test Perplexity: %.3f" % perplexity)


def run_epoch(session, model, data, eval_op, reader, verbose=False):
    epoch_size = ((len(data) // model.config.batch_size) - 1) // model.config.num_steps
    start_time = time.time()
    costs = 0.0
    num_iter = 0
    state = model.initial_state.eval()

    for step, (x_i, x_u, y) in enumerate(reader.data_iterator(data, model.config.batch_size, model.config.num_steps)):
        cost, state, _, logits = session.run([model.cost, model.final_state, eval_op, model._logits],
                                     {model.input_i: x_i,
                                      model.input_u: x_u,
                                      model.targets: y,
                                      model.initial_state: state})
        costs += cost
        num_iter += model.config.num_steps

        if verbose and step % (epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / epoch_size, np.exp(costs / num_iter),
                   num_iter * model.config.batch_size / (time.time() - start_time)))

    return np.exp(costs / num_iter)


def get_setup():
    return MlReader(), "data_sets/src/ml-100k"
    #return LastfmReader(), "data_sets/src/lastfm"


class Config(object):
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 10
    hidden_size = 200  # = embedding_size?
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    item_dim = 1
    user_dim = 1


def test_reader(reader, data_path):
    reader.raw_item_data(data_path)


class LSTMTest(unittest.TestCase):
    @staticmethod
    def test_lstm():
        test_network()

if __name__ == "__main__":
    unittest.main()
