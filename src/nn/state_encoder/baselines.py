# # add two baselines:
# PLD - Pairwise Local Dependency between items (PLD), corresponding to DRR-p in Liu's paper.
# BOI - Bag of Items, corresponding to DRR-u but lack user embeddings.

import tensorflow as tf
import numpy as np
from nn.state_encoder.state_encoder import AbstractStateEncoder

class PLD(AbstractStateEncoder):
    def __init__(self, action_space, state_maxlength, seed=None, config=None):
        super(PLD, self).__init__(action_space, state_maxlength, seed=seed, config=config)
        self.W = tf.Variable(tf.random_normal(shape=[self.action_space, 1], mean=0.0, stddev=0.01, seed=np.random.randint(4096)), name='item_weights', dtype=tf.float32)

    def call(self, input_s, input_f, len_s, s, name='net'): 
        '''
            Here, we also need s (item ids) to get w_i,
            s_t = [q_i1, q_i2, ..., q_it] + [{w_i * q_i * q_j * w_j | i, j = {i_1, i_2, ..., i_t}}]
        '''
        input_state = input_s * input_f # [B L D], combine s and f by dot product
        output1 = tf.reshape(input_state, [-1, input_state.shape[1] * input_state.shape[2]]) # [B, L*D], [q_i1, q_i2, ..., q_it]
        input_w = tf.nn.embedding_lookup(self.W, s) # [B L 1]
        input_w_state = input_w * input_state # [B L D], [w_i1 * q_i1, ..., w_it * q_it]
        output2 = tf.matmul(tf.cast(input_w_state, dtype=tf.float32), tf.cast(tf.transpose(input_w_state, perm=[0, 2, 1]), dtype=tf.float32)) # [B L L]
        output2 = tf.reshape(output2, [-1, output2.shape[1] * output2.shape[2]]) # [B L*L]
        output = tf.concat([output1, output2], axis=1) # [B L*D+L*L]
        output = tf.layers.dense(output, self.action_space, kernel_initializer=tf.random_normal_initializer(seed=np.random.randint(4096)), bias_initializer = tf.constant_initializer(), name=name)
        return output


class BOI(AbstractStateEncoder):
    def __init__(self, action_space, state_maxlength, seed=None, config=None):
        super(BOI, self).__init__(action_space, state_maxlength, seed=seed, config=config)
        self.W = tf.Variable(tf.random_normal(shape=[self.action_space, 1], mean=0.0, stddev=0.01, seed=np.random.randint(4096)), name='item_weights', dtype=tf.float32)

    def call(self, input_s, input_f, len_s, s, name='net'): 
        '''
            Here, we also need s (item ids) to get w_i, 
            s_t = [w_i1*q_i1, ..., w_it*q_it]
        '''
        input_state = input_s * input_f # [B L D], combine s and f by dot product
        input_w = tf.nn.embedding_lookup(self.W, s) # [B L 1]
        input_w_state = input_w * input_state # [B L D], [w_i1 * q_i1, ..., w_it * q_it]
        input_w_state = tf.reshape(input_w_state, [-1, input_w_state.shape[1] * input_w_state.shape[2]]) # [B L*D]
        output = tf.layers.dense(input_w_state, self.action_space, kernel_initializer=tf.random_normal_initializer(seed=np.random.randint(4096)), bias_initializer = tf.constant_initializer(), name=name)
        return output