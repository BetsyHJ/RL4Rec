"""
Attention-based state encoder
Author: Thijs Rood
Contact: thijs.rood@hotmail.com
"""


import numpy as np
import tensorflow as tf
from nn.state_encoder.state_encoder import AbstractStateEncoder

class attention(AbstractStateEncoder):
    def __init__(self, action_space, state_maxlength, seed=None, config=None):
        super(attention, self).__init__(action_space, state_maxlength, seed=seed, config={'rnn_state_dim':128, 'units':128})
        self.rnn_state_dim = config['rnn_state_dim']
        units = config['rnn_state_dim']
        self.W1 = tf.keras.layers.Dense(units, kernel_initializer=tf.random_normal_initializer(seed=np.random.randint(4096)))


    def call(self, input_s, input_f, len_s, name='net'):
        input_state = input_s * input_f # (B L D)
        cell = tf.contrib.rnn.GRUCell(num_units=self.rnn_state_dim, kernel_initializer=tf.random_normal_initializer(seed=np.random.randint(4096)), bias_initializer = tf.constant_initializer())
        h_s, h_t = tf.nn.dynamic_rnn(cell, dtype=tf.float32, sequence_length=len_s, inputs=input_state)

        score = tf.expand_dims(h_t, 1) * self.W1(h_s) # general attention formula
        score = tf.reduce_sum(score, axis=-1) # (B L)

        attention_weights = tf.nn.softmax(score) # (B L)

        # compute the context vector
        context_vector = tf.expand_dims(attention_weights, -1) * h_s # (B L 1) * (B L D)
        context_vector = tf.reduce_sum(context_vector, axis=1) # (B D)

        # simple concatenation
        x = tf.concat([context_vector, h_t], axis=-1)

        # output a fully connected layer
        output = tf.layers.dense(x, self.action_space, kernel_initializer=tf.random_normal_initializer(seed=np.random.randint(4096)), bias_initializer = tf.constant_initializer(), name=name)

        return output, attention_weights
