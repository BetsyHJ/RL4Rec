import tensorflow as tf

class AbstractStateEncoder(tf.layers.Layer):
    def __init__(self, action_space, state_maxlength, config=None):
        super(AbstractStateEncoder, self).__init__()
        self.action_space = action_space
        self.state_maxlength = state_maxlength

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


class MLP(AbstractStateEncoder):
    def __init__(self, action_space, state_maxlength, config=None):
        super(MLP, self).__init__(action_space, state_maxlength)

        
    def call(self, input_s, input_f, len_s, name='net'):
        input_state = self._combine_item_feedback(input_s, input_f, len_s)
        output = tf.layers.dense(input_state, self.action_space, kernel_initializer=w_init, bias_initializer=b_init, name=name)
        return output


class GRU(AbstractStateEncoder):
    def __init__(self, action_space, state_maxlength, config={'rnn_state_dim':128}):
        super(GRU, self).__init__(action_space, state_maxlength)
        self.rnn_state_dim = config['rnn_state_dim']

    def call(self, input_s, input_f, len_s, name='net'):
        input_state = input_s * input_f
        cell_main = tf.contrib.rnn.GRUCell(num_units=self.rnn_state_dim)
        _, h_s = tf.nn.dynamic_rnn(cell_main, dtype=tf.float32, sequence_length=len_s, inputs=input_state)
        output = tf.layers.dense(h_s, self.action_space, name=name)
        return output