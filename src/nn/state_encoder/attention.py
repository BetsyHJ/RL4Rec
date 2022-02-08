"""
Attention-based state encoder
Author: Thijs Rood
Contact: thijs.rood@hotmail.com
Modified by Jin Huang
"""


import numpy as np
import tensorflow as tf
from nn.state_encoder.state_encoder import AbstractStateEncoder

class Attention(AbstractStateEncoder):
    def __init__(self, action_space, state_maxlength, seed=None, config=None):
        super(Attention, self).__init__(action_space, state_maxlength, seed=seed, config=config)
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
        # # # mask the weights for padding vectors, modified by Jin
        # mask = tf.sequence_mask(len_s, self.state_maxlength, dtype=tf.float32) # (B, L)
        # attention_weights = attention_weights * mask # (B, L), with padding 0
        # attention_weights = attention_weights / tf.reduce_sum(attention_weights, axis=-1, keepdims=True) # (B L)

        # compute the context vector
        context_vector = tf.expand_dims(attention_weights, -1) * h_s # (B L 1) * (B L D)
        context_vector = tf.reduce_sum(context_vector, axis=1) # (B D)

        x = tf.concat([context_vector, h_t], axis=-1)

        # output a fully connected layer
        output = tf.layers.dense(x, self.action_space, activation=self.activation, kernel_initializer=tf.random_normal_initializer(seed=np.random.randint(4096)), bias_initializer = tf.constant_initializer(), name=name)

        # return output, attention_weights
        return output
