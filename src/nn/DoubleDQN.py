'''
Author: Jin
Goal: DQN
Following https://www.zhihu.com/people/fuckyou-74/posts
'''

import numpy as np
import tensorflow as tf
# np.random.seed(2020)

class DoubleDQN(object):
    def __init__(self, config):
        self.state_maxlength = config['STATE_MAXLENGTH']
        self.action_space = config['ACTION_SPACE']
        self.action_feature = None
        if 'ACTION_FEATURE' in config:
            self.action_feature = config['ACTION_FEATURE']
            self.action_dim = self.action_feature.shape[1]
        else:
            self.action_dim = config['ACTION_DIM'] # todo: pretrain action feature?
        self.rnn_state_dim = config['RNN_STATE_DIM']
        self.memory_size = config['MEMORY_SIZE']
        self.memory = np.zeros((self.memory_size, self.state_maxlength*4+4), dtype='object')
        self.gamma = config['GAMMA']
        self.lr = config['LEARNING_RATE']
        self.lr_decay_step = int(config['lr_decay_step'])
        self.epsilon_min = config['EPSILON']
        self.epsilon = 0.8
        self.epsilon_decay_step = int(config['epsilon_decay_step'])
        self.batch_size = config['BATCH_SIZE']
        self.learn_step_counter = 0
        self.replace_TargetNet_iter = config['REPLACE_TARGETNET']
        self.optimizer_name = config['OPTIMIZER']
        self.state_encoder = config["state_encoder"]
        self.double_q = True

        self.save_model_file = 'DoubleDQN'
        if 'SAVE_MODEL_FILE' in config:
            self.save_model_file = config['SAVE_MODEL_FILE']
        
        # tf.set_random_seed(2020)
        # init sess
        self._build_net()
        mainNet_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='MainNet')
        TargetNet_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='TargetNet')
        with tf.variable_scope('replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(TargetNet_params, mainNet_params)] # update TargetNet params by using MainNet params

        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        # self.writer = tf.summary.FileWriter("log/logs/", self.sess.graph)
        # exit(0)
        self.with_userinit = False
        print("DQN with user_init:", self.with_userinit)

    def _build_net(self):
        # ---- input ---- 
        self.s = tf.placeholder(tf.int32, [None, self.state_maxlength], name='s') # todo: return s as a representation, not do rnn O(n) everytime 
        self.s_= tf.placeholder(tf.int32, [None, self.state_maxlength], name='s_')
        self.f = tf.placeholder(tf.int32, [None, self.state_maxlength], name='f') 
        self.f_= tf.placeholder(tf.int32, [None, self.state_maxlength], name='f_')
        # self.r = tf.placeholder(tf.float32, [None, ], name='r')
        self.a = tf.placeholder(tf.int32, [None, ], name='a')
        self.q_target = tf.placeholder(tf.float32, [None, ], name='Q_next')
        self.len_s = tf.placeholder(tf.int32, [None, ], name='len_s')
        self.len_s_ = tf.placeholder(tf.int32, [None, ], name='len_s_')
        
        # ----- embedding -----
        if self.action_feature is not None:
            self.action_embeddings = tf.constant(dtype=tf.float32, value=self.action_feature)
        else:
            self.action_embeddings = tf.Variable(tf.random_normal(shape=[self.action_space, self.action_dim], mean=0.0, stddev=0.01), name='action_embeddings', dtype=tf.float32)
        # deal with the masked item:
        self.action_embeddings = tf.concat([self.action_embeddings, tf.constant(np.zeros([1, self.action_dim]), dtype='float32')], axis=0)
        self.feedback_embeddings = tf.Variable(tf.random_normal(shape=[2, self.action_dim], mean=0.0, stddev=0.01), name='feedback_embeddings', dtype=tf.float32)

        # --------- get the feedback embeddings
        self.input_f = tf.nn.embedding_lookup(self.feedback_embeddings, tf.reshape(self.f, [-1]))
        self.input_f_ = tf.nn.embedding_lookup(self.feedback_embeddings, tf.reshape(self.f_, [-1]))
        # -------- be careful: here merge the state (items) with feedback_emb
        self.input_s = tf.reshape(tf.nn.embedding_lookup(self.action_embeddings, tf.reshape(self.s, [-1])) * self.input_f, [-1, self.state_maxlength, self.action_dim]) #(None, d) -> (None, s_lengh, d)
        self.input_s_ = tf.reshape(tf.nn.embedding_lookup(self.action_embeddings, tf.reshape(self.s_, [-1])) * self.input_f_, [-1, self.state_maxlength, self.action_dim]) #(None, d) -> (None, s_lengh, d)

        # # ********* here we use rnn, it can be MLP, CNN ....
        if self.state_encoder.lower() == 'gru':
            w_init, b_init = tf.random_normal_initializer(), tf.constant_initializer()
            # ----- build MainNet -----
            with tf.variable_scope('MainNet'):
                cell_main = tf.contrib.rnn.GRUCell(num_units=self.rnn_state_dim)
                _, h_s = tf.nn.dynamic_rnn(cell_main, dtype=tf.float32, sequence_length=self.len_s, inputs=self.input_s)
                self.q_eval = tf.layers.dense(h_s, self.action_space, kernel_initializer=w_init, bias_initializer=b_init, name='q')
            # ----- build TargetNet -----
            with tf.variable_scope('TargetNet'):
                cell_target = tf.contrib.rnn.GRUCell(num_units=self.rnn_state_dim)
                _, h_s_ = tf.nn.dynamic_rnn(cell_target, dtype=tf.float32, sequence_length=self.len_s_, inputs=self.input_s_)
                self.q_next = tf.layers.dense(h_s_, self.action_space, kernel_initializer=w_init, bias_initializer=b_init, name='t2')

        # ********* here we use 1-layer dense
        elif self.state_encoder.lower() == 'mlp':
            w_init, b_init = tf.random_normal_initializer(), tf.constant_initializer()
            ## before the code is h_s = tf.reduce_mean(self.input_s, axis=1)
            h_s = tf.reduce_sum(self.input_s, axis=1) / tf.reshape(tf.cast(self.len_s, dtype=tf.float32), [-1, 1]) # (None, d)
            h_s_ = tf.reduce_sum(self.input_s_, axis=1) / tf.reshape(tf.cast(self.len_s_, dtype=tf.float32), [-1, 1]) # (None, d)  
            # ----- build MainNet -----
            with tf.variable_scope('MainNet'):
                self.q_eval = tf.layers.dense(h_s, self.action_space, kernel_initializer=w_init, bias_initializer=b_init, name='q_eval')
            # ----- build TargetNet -----
            with tf.variable_scope('TargetNet'):
                self.q_next = tf.layers.dense(h_s_, self.action_space, kernel_initializer=w_init, bias_initializer=b_init, name='q_next')
        else:
            print("error state_encoder")
            exit(1)
        
        # ----- DQN-loss ----- 
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0],dtype=tf.int32), self.a], axis=1)
            self.q_eval_s_a = tf.gather_nd(self.q_eval, a_indices)
        # with tf.variable_scope('q_next'):
        #     self.q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')
        #     self.q_target = tf.stop_gradient(self.q_target)
        
        with tf.variable_scope('loss'):
            # self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_s_a))
            # smooth-l1-loss: https://blog.csdn.net/u014365862/article/details/79924201
            diff = self.q_target - self.q_eval_s_a
            abs_diff = tf.abs(diff)
            smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_diff, 1.)))
            loss = tf.pow(diff, 2) * 0.5 * smoothL1_sign + (abs_diff - 0.5) * (1 - smoothL1_sign)
            self.loss = tf.reduce_mean(loss)

        with tf.variable_scope('optimize'):
            self.global_step = tf.Variable(0, trainable=False)
            self.decayed_lr = tf.train.exponential_decay(self.lr, \
                                        self.global_step, 1, 0.9, staircase=True)
            ## self.decayed_lr = self.lr
            if self.optimizer_name.lower() == 'adam':
                self.optimizer = tf.train.AdamOptimizer(self.decayed_lr).minimize(self.loss)
            elif self.optimizer_name.lower() == 'rmsprop':
                self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.decayed_lr, decay=0.99999).minimize(self.loss)
            else:
                print("Wrong optimizer. Stopped")
                exit(1)
            self.add_global = self.global_step.assign_add(1)
        
    # def _pad_(self, state, max_length, value):
    def _pad_(self, state, value):
        return state + [value] * (self.state_maxlength - len(state))
    # store (s, a, r, s') in Memory
    def store_memory(self, s, a, r, s_): # s = [[item], [feedback]]
        len_s, len_s_ = len(s[0]), len(s_[0])
        if not hasattr(self, 'memory_counter'): # first (s, a, r, s')
            self.memory_counter = 0
        # replace the first input (first in first out)
        index = self.memory_counter % self.memory_size
        state = self._pad_(s[0], self.action_space)
        state_ = self._pad_(s_[0], self.action_space)
        f = self._pad_(s[1], 0) # feedback is 0 or 1 now, it is ok to set 0 or 1, because refering s is [0, 0, 0] so the sum should be 0 too.
        f_ = self._pad_(s_[1], 0)
        self.memory[index, :] = state + state_ + f + f_ + [a, r, len_s, len_s_]
        self.memory_counter += 1
    def store_memorys(self, states, actions, rewards, states_, len_s, len_s_): # s = [[item], [feedback]]
        if (len_s == 0) and (self.with_userinit == False):
            return
        u_batch_size = states.shape[0]
        assert (len_s + 1) == len_s_
        if not hasattr(self, 'memory_counter'): # first (s, a, r, s')
            self.memory_counter = 0
        # replace the first input (first in first out)
        index = self.memory_counter % self.memory_size
        s = states[:, 0]
        f = states[:, 1]
        s_ = states_[:, 0]
        f_ = states_[:, 1]
        r = np.expand_dims(rewards, axis=-1)
        a = np.expand_dims(actions, axis=-1)
        len_s = np.full((u_batch_size, 1), len_s, dtype=np.int32)
        len_s_ = np.full((u_batch_size, 1), len_s_, dtype=np.int32)
        if (index+u_batch_size) > self.memory_size:
            part2_len = index+u_batch_size - self.memory_size
            indices = list(range(index, self.memory_size)) + list(range(part2_len))
            assert len(indices) == u_batch_size
            self.memory[indices, :] = np.hstack((s, s_, f, f_, a, r, len_s, len_s_))
        else:
            self.memory[index:(index+u_batch_size), :] = np.hstack((s, s_, f, f_, a, r, len_s, len_s_))
        self.memory_counter += u_batch_size
        
    def choose_action(self, s, greedy='false'):
        if ((np.random.uniform() < self.epsilon) and (greedy == 'false')) or (len(s[0]) == 0):
            action = np.random.randint(0, self.action_space)
            while action in s[0]:
                action = np.random.randint(0, self.action_space)
        else:
            state = np.array(self._pad_(s[0], self.action_space), dtype=np.int32)
            f = np.array(self._pad_(s[1], 0), dtype=np.int32)
            action_value = self.sess.run(self.q_eval, feed_dict={self.f:f[None, :], self.s:state[None, :], self.len_s:np.array([len(s[0])], dtype=np.int32)})
            action_value = action_value[0, :]
            assert len(action_value) == self.action_space
            action = np.argmax(action_value)
            change_v = np.min(action_value) - 1.0
            while action in s[0]:
                action_value[action] = change_v
                action = np.argmax(action_value)
        return action
    def choose_actions(self, states, len_s, greedy='false'):
        if not hasattr(self, 'interact_count'):
            self.interact_count = 0
        # self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_delta)
        self.interact_count += 1
        # if greedy is false, then do \epsilon-greedy. or just use policy to max
        s = states[:, 0]
        f = states[:, 1]
        batch_size = s.shape[0]
        ## indices for items can not be choosed
        indices_x = np.repeat(range(batch_size), len_s).astype('int')
        indices_y = s[:, :len_s].flatten()

        select_p = np.random.uniform(size=(batch_size, self.action_space))
        select_p[indices_x, indices_y] = -1.
        actions_random = np.argmax(select_p, axis=1)

        # # decrease the epsilon for exploration
        if self.interact_count % self.epsilon_decay_step == 0:
            self.epsilon = max(self.epsilon_min, self.epsilon-0.1)

        if (not self.with_userinit) and (len_s == 0):
            return actions_random
        
        ### use policy
        # --- normal way ----
        action_values = self.sess.run(self.q_eval, feed_dict={self.f:f, self.s:s, self.len_s:np.full((batch_size), len_s, dtype=np.int32)})

        # set the historical action very small value, deal with repeat items
        replace_value = np.min(action_values) - 1.
        action_values[indices_x, indices_y] = replace_value
        actions = np.argmax(action_values, axis=1)
        if greedy.lower() == 'true':
            return actions
        
        return np.where(np.random.uniform(size=(s.shape[0],))<self.epsilon, actions_random, actions)
    
    def rerank(self, s, itemList):
        if len(s[0]) == 0:
            print("to get scores, the state must has something")
            exit(1)
        state = np.array(self._pad_(s[0], self.action_space), dtype=np.int32)
        f = np.array(self._pad_(s[1], 0), dtype=np.int32)
        action_value = self.sess.run(self.q_eval, feed_dict={self.f:f[None, :], self.s:state[None, :], self.len_s:np.array([len(s[0])], dtype=np.int32)})
        action_value = action_value[0, :]
        assert len(action_value) == self.action_space
        scores = action_value[itemList]
        rerankOrder = np.flip(np.argsort(scores))
        return rerankOrder
    def reranks(self, states, len_s, itemLists=None):
        if (self.with_userinit == False) and (len_s == 0):
            print("to get scores, the state must has something")
            exit(1)
        s = states[:, 0]
        f = states[:, 1]
        action_values = self.sess.run(self.q_eval, feed_dict={self.f:f, self.s:s, self.len_s:np.full((s.shape[0]), len_s, dtype=np.int32)})
        assert action_values.shape[1] == self.action_space
        if itemLists == None:
            scores = action_values
            return np.flip(np.argsort(scores)), scores
        else:
            rerankOrder, scores = [], []
            for i in range(itemLists.shape[0]):
                score = action_values[i][itemLists[i]]
                scores.append(score)
                rerankOrder.append(np.flip(np.argsort(score)))
            return rerankOrder, scores

    def learn(self):
        if self.memory_counter < self.batch_size:
            return
        # get batch memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        # print(self.memory_counter, sample_index)
        batch_memory = self.memory[sample_index, :]
        # deal with states with unfixed length
        # states = tf.keras.preprocessing.sequence.pad_sequences(batch_memory[:, 0], \
        #             maxlen=self.state_maxlength, padding='post', value=self.action_space)
        # states_ = tf.keras.preprocessing.sequence.pad_sequences(batch_memory[:, 3], \
        #             maxlen=self.state_maxlength, padding='post', value=self.action_space)
        states = np.array(batch_memory[:, :self.state_maxlength], dtype='int')
        states_ = np.array(batch_memory[:, self.state_maxlength:(self.state_maxlength*2)], dtype='int')
        f = np.array(batch_memory[:, (self.state_maxlength*2):(self.state_maxlength*3)], dtype='int')
        f_ = np.array(batch_memory[:, (self.state_maxlength*3):(self.state_maxlength*4)], dtype='int')
        # learn
        actions = batch_memory[:, -4]
        rewards = batch_memory[:, -3]
        len_s, len_s_ = batch_memory[:, -2], batch_memory[:, -1]
        # print("s, s_:\n", states[:5], '\n', states_[:5], '\n')
        # print("actions, rewards:\n", actions[:5], '\n', rewards[:5], '\n')
        # print("len_s, len_s_", len_s[:5], len_s_[:5])
        q_next, q_eval_next = self.sess.run([self.q_next, self.q_eval], \
                feed_dict={self.s_:states_, self.f_:f_, self.s:states_, self.f:f_, \
                self.len_s:len_s_, self.len_s_:len_s_})

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        if self.double_q:
            max_act4_next=np.argmax(q_eval_next, axis=1)
            selected_q_next = q_next[batch_index, max_act4_next]
        else:
            selected_q_next = np.max(q_next, axis=1)

        q_target = rewards + self.gamma * selected_q_next

        _, loss = self.sess.run([self.optimizer, self.loss], \
                feed_dict={self.s:states, self.f:f, self.a:actions, self.len_s:len_s,self.q_target:q_target})
        # todo: can \epislon can be decreased

        self.learn_step_counter += 1
        # check if change TargetNet params or not
        if self.learn_step_counter % self.replace_TargetNet_iter == 0:
            self.sess.run(self.target_replace_op)
        if self.learn_step_counter % self.lr_decay_step == 0:
            self.lr_decay()
        return loss
    
    def lr_decay(self):
        _, lr = self.sess.run([self.add_global, self.decayed_lr])
        return lr
    def get_lr(self):
        return self.sess.run(self.decayed_lr)

    def load_pretrain_model(self):
        self.saver.restore(self.sess, './checkpoint_dir/' + self.save_model_file)
    def save_model(self):
        self.saver.save(self.sess, './checkpoint_dir/' + self.save_model_file)