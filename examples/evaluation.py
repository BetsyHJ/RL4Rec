import numpy as np
import math

import scipy.sparse as sp
from nn.DQN import DQN_R
from nn.DoubleDQN import DoubleDQN

# seed = np.random.randint(4096)
# print("seed:", seed)
# np.random.seed(seed)

def evaluate(conf, config, sofa, testfile=None):
    if conf['mode'] == 'DQN':
        rl = DQN_R(config)
    elif conf['mode'] == 'DoubleDQN':
        rl = DoubleDQN(config)
    if True:
        np.random.seed(511)
    if testfile is not None:
        test_RatingM = load_test_file(testfile) # scipy.sparse
        users = np.arange(sofa.num_users, dtype=int)
        test_users = []
        for u in users:
            itemList = test_RatingM[u].indices
            if len(itemList) > 0:
                test_users.append(u)
        num_users = len(test_users)
        users = np.array(test_users, dtype=np.int32)
    else:
        # sample K users, K from 1000 ~ 10000
        num_users = sofa.num_users
        users = np.arange(num_users, dtype=int)
    print("the number of test-users is", users.shape)
    np.random.shuffle(users)
    # load RL parameters
    rl.load_pretrain_model()
    # give responds according to GT simulator (SimC)
    batch_size = 1000
    avg_cumulativeReward = []
    cumuC_turn = np.zeros(sofa.episode_length, dtype=np.float32)
    cumu_rewards = []
    u_i_reward = []
    u_rankedi = []

    start_idx = 0
    while start_idx < num_users:
        user_ids = users[start_idx:(start_idx+batch_size)]
        sofa.reset(user_ids)
        cumu_reward = np.zeros(user_ids.shape[0])
        user_batch_size = user_ids.shape[0]
        # print("users are", user_ids)
        while True:
            # to check the policy, now for sofa-single
            if sofa.step_count > 0:
                rerankeditems, scores = rl.reranks(sofa.state, sofa.step_count)
                # print("\nstates are", sofa.state)
                # print('Step_count:', sofa.step_count, rerankeditems)
                # print('the scores are:', scores)
                # print('Step_count', sofa.step_count, 'max-Q value is', np.max(scores))
            actions = rl.choose_actions(sofa.state, sofa.step_count, greedy='true')
            (states_, rewards, done) = sofa.step(actions)
            cumu_reward += rewards
            cumuC_turn[sofa.step_count-1] += np.sum(cumu_reward)
            if done:
                break
        # exit(0)
        cumu_rewards.append(cumu_reward)
        start_idx += user_batch_size
        # statistic Cumulative ratings
        avg_cumulativeReward.append([start_idx, np.mean(np.hstack(cumu_rewards))])
    avg_cumuC_turn = cumuC_turn / num_users
    avg_cumuC_turn = np.concatenate([np.arange(sofa.episode_length).reshape((-1, 1))+1, avg_cumuC_turn.reshape((-1, 1))], axis=1)

    filename = './result/' + config['SAVE_MODEL_FILE'] + '_eval.txt'  # '_eval.txt'
    print("save into file", filename)
    # np.savetxt(filename, np.array(avg_cumulativeReward))
    np.savetxt(filename, np.array(avg_cumuC_turn))
    # np.savetxt(filename, np.array(u_i_reward, dtype='int'), fmt='%d')
    # np.savetxt(filename, np.array(u_rankedi), fmt='%s')
    return

def load_test_file(testfile, fmt="matrix"):
    test_RatingM = sp.csr_matrix(np.loadtxt(testfile).astype('int'))
    print("load test data for single-turn evaluation. Done.")
    return test_RatingM

def eval_yahoo_sinTurn(conf, config, sofa, testfile):
    # load test set
    test_RatingM = load_test_file(testfile)
    if conf['mode'] == 'DQN':
        rl = DQN_R(config)
    elif conf['mode'] == 'DoubleDQN':
        rl = DoubleDQN(config)
    # load RL parameters
    rl.load_pretrain_model()
    # sample K users, K from 1000 ~ 10000
    users = np.arange(sofa.num_users, dtype=int)
    test_users, test_ItemLists, test_ItemRatings = [], [], []
    for u in users:
        itemList = test_RatingM[u].indices
        if len(itemList) > 0:
            test_users.append(u)
            test_ItemLists.append(itemList)
            test_ItemRatings.append(test_RatingM[u].data)
            # assert len(itemList) == 10 # because we have 5% testset as MCAR
    num_users = len(test_users)
    test_users = np.array(test_users, dtype=np.int32)
    test_ItemLists = np.array(test_ItemLists)
    test_ItemRatings = np.array(test_ItemRatings)
    # ++++++++ for each user, interact k turns and then rerank the results ++++++ #
    # k turns interaction
    K = [5] # for now, the max_statelenght is 10
    L = [1, 3, 5, 10]
    repeat_ = 10
    batch_size = 10
    NDCGs_multi = []
    for _ in range(repeat_):
        if True:
            seed = np.random.randint(4096)
            print("seed:", seed)
        NDCGs = []
        start_idx = 0
        while start_idx < num_users:
            user_ids = test_users[start_idx:(start_idx+batch_size)]
            itemList = test_ItemLists[start_idx:(start_idx+batch_size)]
            itemRatings = test_ItemRatings[start_idx:(start_idx+batch_size)]
            sofa.reset(user_ids)
            turn = 0
            while True:
                actions = rl.choose_actions(sofa.state, sofa.step_count, greedy='true')
                (states_, rewards, done) = sofa.step(actions)
                turn += 1
                if turn in K:
                    reOrders, scores = rl.reranks(states_, turn, itemList)
                    # print(len(reOrders), itemList.shape[0])
                    assert itemList.shape[0] == itemRatings.shape[0]
                    assert len(reOrders) == itemList.shape[0]
                    for i in range(itemList.shape[0]):
                        ndcg = []
                        ir = np.where(itemRatings[i]>3, 1., 0.)
                        if np.sum(ir) == 0:
                            continue
                        for l in L:
                            ndcg.append(_get_NDCG(ir[reOrders[i]], l))
                        # print("\nscores:", scores[i])
                        # print("reOrders, reordered_r:", reOrders[i], itemRatings[i][reOrders[i]])
                        # print("ndcg:", ndcg)
                        NDCGs.append(ndcg)
                    # exit(0)
                if done:
                    break
            start_idx += user_ids.shape[0]

        NDCGs = np.array(NDCGs, dtype='float')
        NDCGs = np.mean(NDCGs, axis=0)
        print('\n', '\t'.join(["NDCG@"+str(x) for x in L]), flush=True)
        print('\t'.join([str(round(NDCGs[x], 3)) for x in range(NDCGs.shape[0])]), flush=True)
        NDCGs_multi.append(NDCGs)
    # output the NDCGs_multi
    NDCGs_multi = np.array(NDCGs_multi, dtype=np.float32)
    print("best result: ", np.max(NDCGs_multi, axis=0), ', the corresponding position is', np.argmax(NDCGs_multi, axis=0))
    print("worst result: ", np.min(NDCGs_multi, axis=0), ', the corresponding position is', np.argmin(NDCGs_multi, axis=0))
    print("the average result: ", np.mean(NDCGs_multi, axis=0))

def yahoo_eval_1(conf, config, sofa, testfile):
    ## here we only choose items from testset, except the 1st turn
    # load test set
    test_RatingM = load_test_file(testfile)
    if conf['mode'] == 'DQN':
        rl = DQN_R(config)
    elif conf['mode'] == 'DoubleDQN':
        rl = DoubleDQN(config)
    # load RL parameters
    rl.load_pretrain_model()
    users = np.arange(sofa.num_users, dtype=int)
    test_users, test_ItemLists, test_ItemRatings = [], [], []
    for u in users:
        itemList = test_RatingM[u].indices
        if len(itemList) > 0:
            test_users.append(u)
            test_ItemLists.append(itemList)
            test_ItemRatings.append(test_RatingM[u].data)
            # assert len(itemList) == 10 # because we have 5% testset as MCAR
    num_users = len(test_users)
    test_users = np.array(test_users, dtype=np.int32)
    test_ItemLists = np.array(test_ItemLists)
    test_ItemRatings = np.array(test_ItemRatings)
    # ++++++++ for each user, choose the item from testset, except 1st turn (random-selected) ++++++ #
    # k turns interaction
    K = [5] # for now, the max_statelenght is 10
    L = [1, 3, 5, 10]
    repeat_ = 10
    batch_size = 1000
    avg_cumuC_turn_multi = []
    np.random.seed(20200327)
    for _ in range(repeat_):
        if True:
            seed = np.random.randint(4096)
            print("seed:", seed)
        cumuC_turn = np.zeros(sofa.episode_length, dtype=np.float32)
        start_idx = 0
        while start_idx < num_users:
            user_ids = test_users[start_idx:(start_idx+batch_size)]
            itemList = test_ItemLists[start_idx:(start_idx+batch_size)]
            itemRatings = test_ItemRatings[start_idx:(start_idx+batch_size)]
            sofa.reset(user_ids)
            turn = 0
            cumu_reward = np.zeros(user_ids.shape[0])
            states = sofa.state.copy()
            while True:
                if turn < 0:
                    actions = rl.choose_actions(states, sofa.step_count, greedy='true')
                    (states_, rewards, done) = sofa.step(actions)
                    states = states_.copy()
                else:
                    actions, feedbacks = [], []
                    reOrders, scores = rl.reranks(states, turn, itemList)
                    # print("turns:", turn, "  Q-values are, ", scores, '\n')
                    for i in range(itemList.shape[0]): # for every user
                        if turn >= len(itemList[i]):
                            actions.append(sofa.num_items)
                            feedbacks.append(0)
                            continue
                        reranked_items = itemList[i][reOrders[i]]
                        ratings = itemRatings[i][reOrders[i]]
                        # here, if the first random-select unfortunately choose the item in testset, then bug
                        #### So we do not consider the first randomly selected items, cans = np.isin(reranked_items, list(states[i, 0, 1:]), invert=True)
                        cans = np.isin(reranked_items, list(states[i, 0, :]), invert=True)
                        a = reranked_items[cans][0]
                        actions.append(a)
                        f = 0 if ratings[cans][0] <= 3 else 1
                        feedbacks.append(f)
                    if turn < 10:
                        states[:, 0, turn] = actions
                        states[:, 1, turn] = feedbacks
                    cumu_reward += feedbacks
                    cumuC_turn[turn] += np.sum(cumu_reward)
                    # print(feedbacks, cumuC_turn, '\n')
                    # exit(0)
                turn += 1
                # print(states, '\n')
                if turn >= 10:
                    # exit(0)
                    break
            start_idx += user_ids.shape[0]
        avg_cumuC_turn = cumuC_turn / num_users
        avg_cumuC_turn_multi.append(avg_cumuC_turn)
        print("avg_cumuC_turn:", avg_cumuC_turn, flush=True)
    # output the avg_cumuC_turn_multi
    avg_cumuC_turn_multi = np.array(avg_cumuC_turn_multi, dtype=np.float32)
    print("best result: ", np.max(avg_cumuC_turn_multi, axis=0), ', the corresponding position is', np.argmax(avg_cumuC_turn_multi, axis=0))
    print("worst result: ", np.min(avg_cumuC_turn_multi, axis=0), ', the corresponding position is', np.argmin(avg_cumuC_turn_multi, axis=0))
    print("the average result: ", np.mean(avg_cumuC_turn_multi, axis=0))
    

def _get_NDCG(ratingList, L):
    def _dcg(ratingL):
        dcg_L = 0.0
        for i in range(len(ratingL)):
            dcg_L += (2.0 ** ratingL[i] - 1) / math.log(2.0+i, 2)
        return dcg_L
    true_list = np.flip(np.sort(ratingList))
    # print(ratingList, true_list)
    return _dcg(ratingList[:L]) / _dcg(true_list[:L])


def yahoo_eval_1_calu_itemset(conf, config, sofa, testfile):
    ## here we only choose items from testset, except the 1st turn
    # load test set
    test_RatingM = load_test_file(testfile)
    if conf['mode'] == 'DQN':
        rl = DQN_R(config)
    elif conf['mode'] == 'DoubleDQN':
        rl = DoubleDQN(config)
    # load RL parameters
    rl.load_pretrain_model()
    users = np.arange(sofa.num_users, dtype=int)
    test_users, test_ItemLists, test_ItemRatings = [], [], []
    for u in users:
        itemList = test_RatingM[u].indices
        if len(itemList) > 0:
            test_users.append(u)
            test_ItemLists.append(itemList)
            test_ItemRatings.append(test_RatingM[u].data)
            # assert len(itemList) == 10 # because we have 5% testset as MCAR
    num_users = len(test_users)
    test_users = np.array(test_users, dtype=np.int32)
    batch_size = 10
    np.random.seed(20200327)
    start_idx = 0
    itemsets = []
    while start_idx < num_users:
        user_ids = test_users[start_idx:(start_idx+batch_size)]
        sofa.reset(user_ids)
        states = sofa.state.copy()
        while True:
            actions = rl.choose_actions(states, sofa.step_count, greedy='true')
            itemsets += list(actions)
            (states_, rewards, done) = sofa.step(actions)
            states = states_.copy()
            print('turn', sofa.step_count, states, '\n')
            if done:
                exit(0)
                break
        start_idx += user_ids.shape[0]
    unique, counts = np.unique(itemsets, return_counts=True)
    order = np.flip(np.argsort(counts))
    unique = np.expand_dims(unique[order], axis=1)
    counts = np.expand_dims(counts[order], axis=1)
    filename = conf["data.input.path"] + conf["data.input.dataset"] + '_' + \
        conf["data.gen_model"] + '_' + conf["data.debiasing"] + '_itemset.txt'
    np.savetxt(filename, np.hstack((unique, counts)), fmt='%d')
    print("save into file", filename)