import numpy as np
import tensorflow as tf
from nn.state_encoder.state_encoder import AbstractStateEncoder

class GRU(AbstractStateEncoder):
    def __init__(self, action_space, state_maxlength, seed=None, config={'rnn_state_dim':128}):
        super(GRU, self).__init__(action_space, state_maxlength, seed=seed, config=config)
        self.rnn_state_dim = config['rnn_state_dim']

    def call(self, input_s, input_f, len_s, name='net'):
        input_state = input_s * input_f
        cell = tf.contrib.rnn.GRUCell(num_units=self.rnn_state_dim, kernel_initializer=tf.random_normal_initializer(seed=np.random.randint(4096)), bias_initializer = tf.constant_initializer())
        _, h_s = tf.nn.dynamic_rnn(cell, dtype=tf.float32, sequence_length=len_s, inputs=input_state)
        output = tf.layers.dense(h_s, self.action_space, activation=self.activation, kernel_initializer=tf.random_normal_initializer(seed=np.random.randint(4096)), bias_initializer = tf.constant_initializer(), name=name)
        return output