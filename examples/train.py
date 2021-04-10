import sys
sys.path.append('../src/')
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
from time import time, localtime, strftime

from nn.DQN import DQN_R
from nn.DoubleDQN import DoubleDQN

import matplotlib.pyplot as plt

def train(conf, config, sofa): 
    if conf['mode'] == 'DQN':
        rl = DQN_R(config)
    elif conf['mode'] == 'DoubleDQN':
        rl = DoubleDQN(config)

    num_users = sofa.num_users
    # num_episodes = num_users * 1000
    cumu_rewards = []
    learnstep_cr = []
    load_time, learn_time = 0, 0
    begin_time = time()

    filename = './learning_curves/' + config['SAVE_MODEL_FILE'] + 'plot.pdf'
    plot_data_file = './learning_curves/' + config['SAVE_MODEL_FILE'] + 'plot.data'

    users = np.arange(num_users, dtype=int)
    testuser_only = False
    if testuser_only and conf["data.input.dataset"].lower() == 'yahoo':
        test_file = conf["data.input.path"] + conf["data.input.dataset"] + "_test.ascii"
        RatingM = np.loadtxt(test_file).astype(int)
        users = users[np.where(np.sum(RatingM, axis=1)==0, False, True)]
        print("When learning a policy, we only use the user appeared in the testset. The number is %d" % users.shape[0], flush=True)
        num_users = users.shape[0]

    # ## for remove the non-click users
    # print("*********** Please note we remove the non-click users")
    # clicks = np.where(sofa.ratings > 3, 1, 0)
    # clicks_peruser = np.sum(clicks, axis=1)
    # keepusers = np.where(clicks_peruser<=4, False, True)
    # users = np.arange(num_users, dtype=int)[keepusers]
    # print("********** After removing clicks < 4, we have %d users" % (users.shape[0]))

    step = 0
    batch_size = 10 # 
    epoch = 3000
    trick1_happen = 0
    for epoch in range(epoch):
        np.random.shuffle(users)

        # multi-user interaction
        losses = []
        start_idx = 0
        while start_idx < num_users:
            user_ids = users[start_idx:(start_idx+batch_size)]
            user_batch = user_ids.shape[0]
            start_idx += batch_size
            sofa.reset(user_ids)
            states = sofa.state.copy()
            cumu_reward = np.zeros(user_ids.shape[0])
            states_actions_batch = []
            while True:
                actions = rl.choose_actions(states, sofa.step_count)
                (states_, rewards, done) = sofa.step(actions)
                # print(states.shape, states_.shape)
                cumu_reward += rewards
                states_actions_batch.append((states.copy(), actions, rewards, states_.copy()))
                rl.store_memorys(states, actions, np.where(rewards==0, -1, 2), states_, sofa.step_count-1, sofa.step_count)
                # rl.store_memorys(states, actions, rewards, states_, sofa.step_count-1, sofa.step_count)
                states = states_.copy()
                if done:
                    break
            # if rl.epsilon == rl.epsilon_min:
            #     keep = np.where(cumu_reward<1, False, True)
            #     for turn in range(len(states_actions_batch)):
            #         (s, a, r, s_) = states_actions_batch[turn]
            #         trick1_happen += (user_batch - np.sum(keep))
            #         if np.sum(keep) == 0:
            #             continue
            #         rl.store_memorys(s[keep], a[keep], np.where(r==0, -1, 2)[keep], s_[keep], turn, turn+1)
            # else: 
            #     for turn in range(len(states_actions_batch)):
            #         (s, a, r, s_) = states_actions_batch[turn]
            #         rl.store_memorys(s, a, np.where(r==0, -1, 2), s_, turn, turn+1)
            cumu_rewards += list(cumu_reward)
            for _ in range(4):
                loss = rl.learn()
                if loss is None:
                    break
                # print(loss)
                losses.append(loss)

            if rl.learn_step_counter == 0:
                continue
            if rl.learn_step_counter % 1000 == 0:
                # print(cumu_rewards)
                avg_cumulativeReward = np.mean(cumu_rewards)
                cumu_rewards = []
                learnstep_cr.append([rl.learn_step_counter, avg_cumulativeReward])
                if (len(learnstep_cr) % 10 == 0):
                    plot_cumuReward(learnstep_cr, filename, plot_data_file)
            ## save model
            if rl.learn_step_counter % 5000 == 0:
                print("\nlearning step : %d, time consuming %4e s" % (rl.learn_step_counter, time()-begin_time), flush=True)
                print("learning step: %d, lr : %.5e, epsilon: %2f" % (rl.learn_step_counter, rl.get_lr(), rl.epsilon))
                print("loss:%.3e" % (np.mean(losses)), flush=True)
                begin_time = time()
                rl.save_model()
    # if (epoch + 1) % 100 == 0:
    #     print("Till epoch %d, the trick-1 works for %d times" % (epoch+1, trick1_happen))

    plot_cumuReward(learnstep_cr, filename, plot_data_file)
    print("plot save into file:%s" % filename)
    
def plot_cumuReward(learnstep_cr, filename='line_plot.pdf', plot_data_file='line_plot.data'):
    np.savetxt(plot_data_file, np.array(learnstep_cr))
    # print("plot save into file:%s" % filename )
    plt.clf()
    # fig = plt.figure(figuresize=4, 5)
    learnstep_cr = np.array(learnstep_cr)
    plt.plot(learnstep_cr[:, 0].astype('int'), learnstep_cr[:, 1].astype('float'))
    plt.ylabel('Cumulative_Click')
    plt.xlabel('training steps')
    plt.savefig(filename)
    # plt.show(fig)
    plt.show(block=False)
    plt.close()
