import tensorflow as tf
import numpy as np

class AbstractStateEncoder(tf.layers.Layer):
    def __init__(self, action_space, state_maxlength, seed=None, config=None):
        super(AbstractStateEncoder, self).__init__()
        self.action_space = action_space
        self.state_maxlength = state_maxlength
        # self.seed = seed

    def build(self, input_shape):  # TensorShape of input when run call(), inference from inputs
        self.built = True

    def _combine_item_feedback(self, input_s, input_f, len_s):
        # # concat input_s and input_f first by dot
        input_state = input_s * input_f # [B L D]
        input_state = tf.reduce_sum(input_state, axis=1) / tf.reshape(tf.cast(len_s, dtype=tf.float32), [-1, 1]) # [B D]
        return input_state

    def call(self, input_s, input_f, training=None):
        ''' 
        Input:
            input_s: the embedding of the list of historical interacted items [v_i1, v_i2, ..., v_in], shape [B L D]
            input_f: the embedding of the list of user feedback [v_f1, v_f2, ..., v_in], shape [B L D]
        '''
        pass
