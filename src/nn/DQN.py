'''
Author: Jin
Goal: DQN
Following https://www.zhihu.com/people/fuckyou-74/posts
'''

import sys
# sys.path.append('./')
from nn.state_encoder.gru import GRU
from nn.state_encoder.mlp import MLP
from nn.state_encoder.attention import Attention
from nn.state_encoder.attention_liu import Att_by_Liu
from nn.state_encoder.cnn import CNN
from nn.state_encoder.baselines import PLD, BOI

import numpy as np
import tensorflow as tf
# tf.compat.v1.disable_eager_execution()

class DQN_R(object):
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
        # self.mini_batch = 64
        self.learn_step_counter = 0
        self.replace_TargetNet_iter = config['REPLACE_TARGETNET']
        self.optimizer_name = config['OPTIMIZER']
        self.state_encoder = config["state_encoder"]
        self.num_reward_type = 2
        if self.state_encoder.lower() not in ['pld', 'boi']:
            self.activation = config['activation']

        self.save_model_file = 'DQN'
        if 'SAVE_MODEL_FILE' in config:
            self.save_model_file = config['SAVE_MODEL_FILE']
        
        self.with_userinit = False
        print("DQN with user_init:", self.with_userinit)

        # init sess
        tf.reset_default_graph()
        self._build_net()
        mainNet_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='MainNet')
        TargetNet_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='TargetNet')
        with tf.variable_scope('replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(TargetNet_params, mainNet_params)] # update TargetNet params by using MainNet params

        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(tf.global_variables_initializer())
        # tensorboard --logdir ./logs/ --host=127.0.0.1
        # self.writer = tf.summary.FileWriter("./logs/", self.sess.graph)
        # for var in tf.trainable_variables():
        #     print(var.name, ':', self.sess.run(var).shape)
        # exit(1)
        self.saver = tf.train.Saver()

        # self.epsilon_delta = (self.epsilon - self.epsilon_min) / 50000.0

        # # training according to the previous training
        # keep_training = True
        # if keep_training:
        #     self.load_pretrain_model()
        #     self.save_model_file += "_keeping_"
        #     print("continue learning. And save checkpoint into file:", self.save_model_file)
        #     self.epsilon = 0.1

    def _build_net(self):
        # ---- input ---- 
        self.s = tf.placeholder(tf.int32, [None, self.state_maxlength], name='s') # todo: return s as a representation, not do rnn O(n) everytime 
        self.s_= tf.placeholder(tf.int32, [None, self.state_maxlength], name='s_')
        self.f = tf.placeholder(tf.int32, [None, self.state_maxlength], name='f') 
        self.f_= tf.placeholder(tf.int32, [None, self.state_maxlength], name='f_')
        self.r = tf.placeholder(tf.float32, [None, ], name='r')
        self.a = tf.placeholder(tf.int32, [None, ], name='a')
        self.len_s = tf.placeholder(tf.int32, [None, ], name='len_s')
        self.len_s_ = tf.placeholder(tf.int32, [None, ], name='len_s_')
        # --- normal way ---
        s, s_, f, f_, r, a, len_s, len_s_ = self.s, self.s_, self.f, self.f_, self.r, self.a, self.len_s, self.len_s_

        # # --- input with Dataset way ----
        # self.mini_batch = tf.placeholder(tf.int64)
        # dataset = tf.data.Dataset.from_tensor_slices((self.s, self.s_, self.f, self.f_, self.r, self.a, self.len_s, self.len_s_)).batch(self.mini_batch)#.repeat() #repeat the data
        # self.data_iter = dataset.make_initializable_iterator()
        # s, s_, f, f_, r, a, len_s, len_s_ = self.data_iter.get_next()

        # ----- embedding -----
        if self.action_feature is not None:
            self.action_embeddings = tf.constant(dtype=tf.float32, value=self.action_feature)
        else:
            self.action_embeddings = tf.Variable(tf.random_normal(shape=[self.action_space, self.action_dim], mean=0.0, stddev=0.01, seed=np.random.randint(4096)), name='action_embeddings', dtype=tf.float32)
        # deal with the masked item:
        self.action_embeddings = tf.concat([self.action_embeddings, tf.constant(np.zeros([1, self.action_dim]), dtype='float32')], axis=0)
        self.feedback_embeddings = tf.Variable(tf.random_normal(shape=[self.num_reward_type, self.action_dim], mean=0.0, stddev=0.01, seed=np.random.randint(4096)), name='feedback_embeddings', dtype=tf.float32)
        # for initial user embedding, shared by all the users
        if self.with_userinit:
            init_user_vec = tf.Variable(tf.random_normal(shape=[1, self.action_dim], mean=0.0, stddev=0.01, seed=np.random.randint(4096)), name='init_user_vec', dtype=tf.float32)
            self.action_embeddings = tf.concat([self.action_embeddings, init_user_vec], axis=0)
        self.feedback_embeddings = tf.concat([self.feedback_embeddings, tf.constant(np.ones([1, self.action_dim]), dtype='float32')], axis=0)

        '''
        Use state encoder class
        '''
        # --------- get the feedback embeddings [B L D]
        self.input_f = tf.nn.embedding_lookup(self.feedback_embeddings, f)
        self.input_f_ = tf.nn.embedding_lookup(self.feedback_embeddings, f_)
        # --------- get the items embeddings [B L D]
        self.input_s = tf.nn.embedding_lookup(self.action_embeddings, s)
        self.input_s_ = tf.nn.embedding_lookup(self.action_embeddings, s_)
        self.eee = self.input_s * self.input_f
        config = {}
        
        if self.state_encoder.lower() not in ['pld', 'boi']:
            config['activation'] = self.activation
        if self.state_encoder.lower() == 'mlp':
            state_encoder = MLP
            print("Mode MLP: 1-layer MLP with activation", config['activation'])
        elif self.state_encoder.lower() == 'gru':
            state_encoder = GRU
            config['rnn_state_dim'] = self.rnn_state_dim
            print("Mode GRU: 1-layer GRU and 1-layer dense")
        elif self.state_encoder.lower() == 'att':
            state_encoder = Attention
            config['rnn_state_dim'] = self.rnn_state_dim
            config['unit'] = self.rnn_state_dim
            assert config['unit'] == config['rnn_state_dim'] # since unit is the dim of W used to compute attention weights
            print("Mode Attention: 1-layer GRU, attention layer and one output (dense) layer")
        elif self.state_encoder.lower() == 'att_liu':
            state_encoder = Att_by_Liu
            print("Mode Att_Liu: Dense, attention layer and one output (dense) layer")
        elif self.state_encoder.lower() == 'cnn':
            state_encoder = CNN
            config['cnn_state_dim'] = self.rnn_state_dim
            print("Mode CNN: 1-layer CNN and one output (dense) layer")
        elif self.state_encoder.lower() == 'pld':
            state_encoder = PLD
            print("Model PLD: one output layer")
        elif self.state_encoder.lower() == 'boi':
            state_encoder = PLD
            print("Model BOI: one output layer")

        # # Two networks, when learn_step_counter == 0, do replace to make initialization of two networks the same.
        # ----- build MainNet -----
        with tf.variable_scope('MainNet'):
            if self.state_encoder.lower() in ['pld', 'boi']:
                self.q_eval = state_encoder(self.action_space, self.state_maxlength, config=config)(self.input_s, self.input_f, self.len_s, s, name='q')
            else:
                self.q_eval = state_encoder(self.action_space, self.state_maxlength, config=config)(self.input_s, self.input_f, self.len_s, name='q')
        # ----- build TargetNet -----
        with tf.variable_scope('TargetNet'):
            if self.state_encoder.lower() in ['pld', 'boi']:
                self.q_next = state_encoder(self.action_space, self.state_maxlength, config=config)(self.input_s_, self.input_f_, self.len_s_, s_, name='t2')
            else:
                self.q_next = state_encoder(self.action_space, self.state_maxlength, config=config)(self.input_s_, self.input_f_, self.len_s_, name='t2')
        
        # # ----- DQN-loss ----- 
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(a)[0],dtype=tf.int32), a], axis=1)
            self.q_eval_s_a = tf.gather_nd(self.q_eval, a_indices)
        with tf.variable_scope('q_next'):
            self.q_target = r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_') #* (1.-done) + self.gamma * external_rewards
            self.q_target = tf.stop_gradient(self.q_target)
        
        with tf.variable_scope('loss'):
            # # l2-loss
            # self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_s_a))
            # # smooth-l1-loss: https://blog.csdn.net/u014365862/article/details/79924201
            diff = self.q_target - self.q_eval_s_a
            abs_diff = tf.abs(diff)
            smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_diff, 1.)))
            loss = tf.pow(diff, 2) * 0.5 * smoothL1_sign + (abs_diff - 0.5) * (1 - smoothL1_sign)
            self.loss = tf.reduce_mean(loss)



        with tf.variable_scope('optimize'):
            self.global_step = tf.Variable(0, trainable=False)
            self.decayed_lr = tf.train.exponential_decay(self.lr, \
                                        self.global_step, 1, 0.9, staircase=True)
            # self.decayed_lr = self.lr
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
        # print("index", indices_x, indices_y)
        # print("value", action_values)

        # # random select
        select_p = np.random.uniform(size=(batch_size, self.action_space))
        # select_p[indices_x, indices_y] = -1.
        actions_random = np.argmax(select_p, axis=1)

        # # decrease the epsilon for exploration
        if self.interact_count % self.epsilon_decay_step == 0:
            self.epsilon = max(self.epsilon_min, self.epsilon-0.1)

        if (not self.with_userinit) and (len_s == 0):
            return actions_random
        
        # # choose actions
        # --- normal way ----
        action_values = self.sess.run(self.q_eval, feed_dict={self.f:f, self.s:s, self.len_s:np.full((batch_size), len_s, dtype=np.int32)})
        # # human intervention: set the historical action very small value, deal with repeat items
        # replace_value = np.min(action_values) - 1.
        # action_values[indices_x, indices_y] = replace_value
        actions = np.argmax(action_values, axis=1)
        if greedy.lower() == 'true':
            return actions
        
        # # do \epsilon-greedy
        return np.where(np.random.uniform(size=(s.shape[0],))<self.epsilon, actions_random, actions)


    def reranks(self, states, len_s, itemLists=None):
        if (self.with_userinit == False) and (len_s == 0):
            ## randomly
            if itemLists is None:
                scores = np.random.uniform(size=(states[:, 0].shape[0], self.action_space))
                return np.flip(np.argsort(scores), axis=-1), scores
            else:
                rerankOrder, scores = [], []
                for i in range(itemLists.shape[0]):
                    score = np.random.uniform(size=(len(itemLists[i])))
                    scores.append(score)
                    rerankOrder.append(np.flip(np.argsort(score)))
                return rerankOrder, scores
            # print("to get scores, the state must has something")
            # exit(1)
        s = states[:, 0]
        f = states[:, 1]
        action_values = self.sess.run(self.q_eval, feed_dict={self.f:f, self.s:s, self.len_s:np.full((s.shape[0]), len_s, dtype=np.int32)})
        assert action_values.shape[1] == self.action_space
        if itemLists is None:
            scores = action_values
            return np.flip(np.argsort(scores), axis=-1), scores
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
        # check if change TargetNet params or not
        if self.learn_step_counter % self.replace_TargetNet_iter == 0: # at first, behavior and target net are same.
            self.sess.run(self.target_replace_op)
        # get batch memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        # print(self.memory_counter, sample_index)
        batch_memory = self.memory[sample_index, :]
        states = np.array(batch_memory[:, :self.state_maxlength], dtype='int')
        states_ = np.array(batch_memory[:, self.state_maxlength:(self.state_maxlength*2)], dtype='int')
        f = np.array(batch_memory[:, (self.state_maxlength*2):(self.state_maxlength*3)], dtype='int')
        f_ = np.array(batch_memory[:, (self.state_maxlength*3):(self.state_maxlength*4)], dtype='int')

        feed_dict={}
        feed_dict[self.s] = states
        feed_dict[self.a] = batch_memory[:, -4]
        feed_dict[self.r] = batch_memory[:, -3]
        feed_dict[self.s_] = states_
        feed_dict[self.f] = f
        feed_dict[self.f_] = f_
        feed_dict[self.len_s] = batch_memory[:, -2]
        feed_dict[self.len_s_] = batch_memory[:, -1]
        # # ---------- learning & update ---------#
        # # learn in one batch
        # print(self.sess.run(self.q_next, feed_dict=feed_dict))
        _, loss = self.sess.run([self.optimizer, self.loss], feed_dict=feed_dict)
        
        self.learn_step_counter += 1
        if self.learn_step_counter % self.lr_decay_step == 0:
            self.lr_decay()
        return loss
    
    def lr_decay(self):
        _, lr = self.sess.run([self.add_global, self.decayed_lr])
        return lr
    def get_lr(self):
        return self.sess.run(self.decayed_lr)

    def load_pretrain_model(self):
        # print('./checkpoint_dir/' + self.save_model_file)
        self.saver.restore(self.sess, './checkpoint_dir/' + self.save_model_file)
    def save_model(self):
        self.saver.save(self.sess, './checkpoint_dir/' + self.save_model_file)