import tensorflow as tf
import numpy as np
from nn.state_encoder.state_encoder import AbstractStateEncoder

class MLP(AbstractStateEncoder):
    def __init__(self, action_space, state_maxlength, seed=None, config=None):
        super(MLP, self).__init__(action_space, state_maxlength, seed=seed, config=config)

    def call(self, input_s, input_f, len_s, name='net'):
        input_state = self._combine_item_feedback(input_s, input_f, len_s)
        output = tf.layers.dense(input_state, self.action_space, kernel_initializer=tf.random_normal_initializer(seed=np.random.randint(4096)), bias_initializer = tf.constant_initializer(), name=name)
        return output


