import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import deque
import copy
import tensorflow as tf

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.compat.v1.reset_default_graph()
from tensorflow import optimizers

def matching(aep_fuse, d, match_strategy, args):
    if match_strategy == "greedy":
        greedy(aep_fuse, d, args)
    elif match_strategy == "hun":
        hun(aep_fuse, d, args)
    elif match_strategy == "sm":
        smat(aep_fuse, d, args)
    elif match_strategy == "rl":
        ceaff_rl(aep_fuse, d, args)

#### greedy
def greedy(aep_fuse, d, args):
    if args.mode == "mul":
        mul_max(1 - aep_fuse, d.test_lefts, d.test_rights, d.test)
    elif args.mode == "unm":
        eva_unm(aep_fuse)
        eva_unm_tbnns(aep_fuse)
    else:
        eva(aep_fuse)
        # eva_print(aep_fuse, ouf)

def mul_max(dis, test_lefts, test_rights, test):
    coun = 0
    for i in range(dis.shape[0]):
        rank = dis[i, :].argsort()
        chosen = rank[0]
        if (test_lefts[i], test_rights[chosen]) in test:
            coun += 1
    print(len(test))
    print(coun)
    precision = coun*1.0/len(dis)
    recall = coun*1.0/len(test)
    f1 = 2/(1/precision + 1/recall)
    print(str(precision) + "\t" + str(recall)+ "\t" + str(f1))

def eva_unm(sim_mat, use_min = False):
    if use_min is True:
        predicted = np.argmin(sim_mat, axis=1)
    else:
        predicted = np.argmax(sim_mat, axis=1)
    cor = predicted == np.array(range(sim_mat.shape[0]))
    cor_num = np.sum(cor)
    precision = cor_num * 1.0 / len(cor)
    recall = cor_num * 1.0 / 10500
    f1 = 2 / (1 / precision + 1 / recall)
    print(cor_num)
    print(str(precision) + "\t" + str(recall) + "\t" + str(f1))

def eva_unm_tbnns(sim):
    detect_cnt = 0
    cor_cnt = 0
    for i in range(sim.shape[0]):
        rank = (-sim[i, :]).argsort()
        minrank = rank[0]
        rank_col = (-sim[:, minrank]).argsort()
        minrank_col = rank_col[0]
        if minrank_col == i:
            detect_cnt += 1
            if i == minrank:
                cor_cnt += 1
    precision = cor_cnt * 1.0 / detect_cnt
    recall = cor_cnt * 1.0 / 10500
    f1 = 2 / (1 / precision + 1 / recall)
    print(str(cor_cnt) + "\t" + str(detect_cnt))
    print(str(precision) + "\t" + str(recall) + "\t" + str(f1))

def eva(sim_mat, use_min = False):
    if use_min is True:
        predicted = np.argmin(sim_mat, axis=1)
    else:
        predicted = np.argmax(sim_mat, axis=1)
    cor = predicted == np.array(range(sim_mat.shape[0]))
    cor_num = np.sum(cor)
    print("Acc: " + str(cor_num) + ' / ' + str(len(cor)) + ' = ' + str(cor_num*1.0/len(cor)))

def eva_print(sim_mat, ouf, use_min = False):
    if use_min is True:
        predicted = np.argmin(sim_mat, axis=1)
    else:
        predicted = np.argmax(sim_mat, axis=1)
    cor = predicted == np.array(range(sim_mat.shape[0]))
    cor_num = np.sum(cor)
    print("Acc: " + str(cor_num) + ' / ' + str(len(cor)) + ' = ' + str(cor_num*1.0/len(cor)))
    for i in range(len(cor)):
        ouf.write(str(i) + '\t' + str(predicted[i]) + '\n')
    # print the alignnmet results

#### hungarian
def hun(aep_fuse, d, args):
    if args.mode == "unm":
        aep_fuse = np.pad(aep_fuse, [(0, 0), (0, aep_fuse.shape[0] - aep_fuse.shape[1])], mode='constant')
        assert aep_fuse.shape[0] == aep_fuse.shape[1]
    cost = 1 - aep_fuse
    del aep_fuse
    ent_num = len(cost)
    row_ind, col_ind = linear_sum_assignment(cost)
    trueC = 0
    if args.mode == "mul":
        for i in range(len(row_ind)):
            if (d.test_lefts[row_ind[i]], d.test_rights[col_ind[i]]) in d.test:
                trueC += 1
        print(trueC)
        precision = trueC * 1.0 / ent_num
        recall = trueC * 1.0 / len(d.test)
        f1 = 2 / (1 / precision + 1 / recall)
        print(str(precision) + "\t" + str(recall) + "\t" + str(f1))
    elif args.mode == "unm":
        for i in range(len(row_ind)):
            if row_ind[i] < 10500:
                if row_ind[i] == col_ind[i]:
                    trueC += 1
        precision = trueC * 1.0 / 10500
        recall = trueC * 1.0 / 10500
        f1 = 2 / (1 / precision + 1 / recall)
        print(len(row_ind))
        print(trueC)
        print(str(precision) + "\t" + str(recall) + "\t" + str(f1))
    else:
        for i in range(len(row_ind)):
            if row_ind[i] == col_ind[i]:
                trueC += 1
            # write the results!!!
            # ouf.write(str(row_ind[i]) + '\t' + str(col_ind[i]) + '\n')
        print("Acc: " + str(trueC) + ' / ' + str(ent_num) + ' = ' + str(trueC * 1.0 / ent_num))

########SMAT
def smat(aep_fuse, d, args):
    if args.mode == "unm":
        aep_fuse = np.pad(aep_fuse, [(0, 0), (0, aep_fuse.shape[0] - aep_fuse.shape[1])], mode='constant')
        assert aep_fuse.shape[0] == aep_fuse.shape[1]
    gapp = 105000
    scale = aep_fuse.shape[0]
    scale_1 = aep_fuse.shape[1]
    MALE_PREFS = {}
    FEMALE_PREFS = {}

    def get_pref(sim, dim=1):
        pref = np.argsort(sim, axis=dim)
        return pref
    pref = get_pref(-aep_fuse)
    pref_col = get_pref(-aep_fuse, dim=0)
    del aep_fuse
    # print("Generate the preference scores time elapsed: {:.4f} s".format(time.time() - t))
    def getmale(pref):
        for i in range(scale):
            lis = (pref[i] + gapp).tolist()
            MALE_PREFS[i] = lis
        return MALE_PREFS
    MALE_PREFS = getmale(pref)
    del pref
    # print("Forming the preference scores time 1 elapsed: {:.4f} s".format(time.time() - t))
    for i in range(scale_1):
        FEMALE_PREFS[i + gapp] = pref_col[:, i].tolist()
    del pref_col
    # print("Forming the preference scores time 2 elapsed: {:.4f} s".format(time.time() - t))
    matches = gale_shapley(set(range(scale)), MALE_PREFS, FEMALE_PREFS)
    del MALE_PREFS
    del FEMALE_PREFS
    # print("Deferred acceptance time elapsed: {:.4f} s".format(time.time() - t))
    trueC = 0
    if args.mode == "mul":
        for match in matches:
            if (d.test_lefts[int(match[0])], d.test_rights[int(match[1]) - gapp]) in d.test:
                trueC += 1
        print(trueC)
        print(len(matches))
        print(len(d.test))
        precision = trueC * 1.0 / len(matches)
        recall = trueC * 1.0 / len(d.test)
        f1 = 2 / (1 / precision + 1 / recall)
        print(str(precision) + "\t" + str(recall) + "\t" + str(f1))
    else:
        for match in matches:
            if int(match[0]) + gapp == int(match[1]):
                trueC += 1
            # write the results!!!
            # ouf.write(str(int(match[0])) + '\t' + str(int(match[1]) - gapp) + '\n')
        if args.mode == "unm":
            precision = trueC * 1.0 / 10500
        else:
            precision = trueC * 1.0 / 10500
        recall = trueC * 1.0 / 10500
        f1 = 2 / (1 / precision + 1 / recall)
        print(len(matches))
        print(trueC)
        print(str(precision) + "\t" + str(recall) + "\t" + str(f1))

def pref_to_rank(pref):
    return {
        a: {b: idx for idx, b in enumerate(a_pref)}
        for a, a_pref in pref.items()
    }

def gale_shapley(A, A_pref, B_pref):
    """Create a stable matching using the
    Gale-Shapley algorithm.

    A -- set[str].
    B -- set[str].
    A_pref -- dict[str, list[str]].
    B_pref -- dict[str, list[str]].

    Output: list of (a, b) pairs.
    """
    B_rank = pref_to_rank(B_pref)
    del B_pref
    ask_list = {a: deque(bs) for a, bs in A_pref.items()}
    del A_pref
    pair = {}
    remaining_A = set(A)
    while len(remaining_A) > 0:
        a = remaining_A.pop()
        b = ask_list[a].popleft()
        if b not in pair:
            pair[b] = a
        else:
            a0 = pair[b]
            b_prefer_a0 = B_rank[b][a0] < B_rank[b][a]
            if b_prefer_a0:
                remaining_A.add(a)
            else:
                remaining_A.add(a0)
                pair[b] = a
    return [(a, b) for b, a in pair.items()]

########RL
def ite1(aep_fuse, aep_fuse_r, dic_row, dic_col, mmtached, mode, test):
    # print(aep_fuse.shape)
    aep_fuse_rank = copy.deepcopy(aep_fuse)
    results_stru = aep_fuse_rank.argsort(axis = 1)[:,0]
    aep_fuse_rank.sort(axis = 1)
    left2right = {i: results_stru[i] for i in range(len(results_stru))}
    aep_fuse_rank_r = copy.deepcopy(aep_fuse_r)
    results_stru_r = aep_fuse_rank_r.argsort(axis = 1)[:,0]
    aep_fuse_rank_r.sort(axis = 1)
    right2left = {i: results_stru_r[i] for i in range(len(results_stru_r))}
    confident = dict()
    row = []
    col = []
    for item in left2right:
        if right2left[left2right[item]] == item:
            confident[item] = left2right[item]
            row.append(item)
            col.append(left2right[item])
    # print('Confi in fuse: ' + str(len(confident)))
    correct = 0
    if mode == 'mul' or mode == 'unm':
        for i in confident:
            mmtached[dic_row[i]]=dic_col[confident[i]]
            if (dic_row[i], dic_col[confident[i]]) in test:
                correct += 1
    else:
        for i in confident:
            mmtached[dic_row[i]]=dic_col[confident[i]]
            if dic_col[confident[i]] == dic_row[i]:
                correct += 1
    # print('Correct in fuse: ' + str(correct))
    # after removal, need to define a mapping function to map column/rows indexes to the origional indexes
    newind_row = 0
    new2old_row = dict()
    newind_col = 0
    new2old_col = dict()
    for item in range(aep_fuse.shape[0]):
        if item not in row:
            new2old_row[newind_row] = dic_row[item] # dic_row item not just item # item is one-hop map while dic_row...
            newind_row += 1
    for item in range(aep_fuse.shape[1]):
        if item not in col:
            new2old_col[newind_col] = dic_col[item]
            newind_col += 1
    aep_fuse_new = np.delete(aep_fuse, row, axis=0)
    aep_fuse_new = np.delete(aep_fuse_new, col, axis=1)
    aep_fuse_r_new = aep_fuse_new.T
    return aep_fuse_new, aep_fuse_r_new, len(confident), correct, new2old_row, new2old_col, mmtached

def evarerank(aep_fuse_new, new2old_row, new2old_col):
    aep_fuse = aep_fuse_new
    del aep_fuse_new
    # print(aep_fuse.shape)
    aep_rank = copy.deepcopy(aep_fuse)
    del aep_fuse
    aep_rank = -aep_rank
    results = aep_rank.argsort(axis=1)[:, 0]
    correct = 0
    leftids = []
    pairs = []
    for i in range(len(new2old_row)):
        leftid = new2old_row[i]
        leftids.append(leftid)
        rightid = new2old_col[results[i]]
        if leftid == rightid:
            correct += 1
        pairs.append([leftid, rightid])
    # print("maximum strategy " + str(correct))
    # print()
    mul = dict()
    for item in pairs:
        if item[1] not in mul:
            mul[item[1]] = 1
        else:
            mul[item[1]] += 1
    multiple = []
    mulsource = 0
    for item in mul:
        if mul[item] > 1:
            multiple.append([item,mul[item]])
            mulsource += mul[item]
    # print("multiple source entities are aligned to the same target entities!")
    # print('source ' + str(mulsource))
    # print('target ' + str(len(multiple)))
    # print()
    rightids = []
    for i in range(len(new2old_col)):
        rightid = new2old_col[i]
        rightids.append(rightid)
    count = 0
    for left in leftids:
        if left in rightids:
            count += 1
    # print("total " + str(count))
    top_ids = aep_rank.argsort(axis=1)[:, :10]
    wrongsx = []
    wrongsy = []
    truePositions = []
    top10 = 0
    for ii in range(len(leftids)):
        pos = np.where(np.array(rightids) == leftids[ii])[0]
        if len(pos) > 0:
            pos = pos[0]
            truePositions.append(pos)
        else:
            pos = -1
        if pos in top_ids[ii]:
            top10 += 1
            if pos != top_ids[ii][0]:
                wrongsx.append(ii)
                wrongsy.append(pos)
    # print("top10 " + str(top10))
    aep_rank.sort(axis=1)
    top_scores = -aep_rank[:, :10]
    scores = -aep_rank[:, 0]
    newindex = (-scores).argsort()
    cans = top_ids[newindex, :]
    scores = top_scores[newindex, :]
    return leftids, rightids, newindex, cans, scores

class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess
        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error
        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=n_features,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )
            self.acts_prob = tf.layers.dense(
                inputs=l1,
                units=n_actions,    # output units
                activation=tf.nn.softmax,   # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )
        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)
    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v
    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})   # get probabilities for all actions
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   # return a int

class Critic(object):
    def __init__(self, sess, GAMMA, n_features, lr=0.01):
        self.sess = sess
        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')
        self.GAMMA = GAMMA
        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=n_features,  # number of hidden units
                activation=tf.nn.relu,  # None
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )
            self.v = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )
        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + self.GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)
    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]
        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                          {self.s: s, self.v_: v_, self.r: r})
        return td_error

def ceaff_rl(aep_fuse, d, args):
    if args.mode == "mul" or args.mode == "unm":
        dic_row = {i: d.test_lefts[i] for i in range(len(d.test_lefts))}  # test_lefts
        dic_col = {i: d.test_rights[i] for i in range(len(d.test_rights))}  # test_rights
    else:
        dic_row = {i: i for i in range(len(d.test))}
        dic_col = {i: i for i in range(len(d.test))}
    total_matched = 0
    total_true = 0
    ent_num = len(aep_fuse)

    aep_fuse_new = copy.deepcopy(1 - aep_fuse)
    aep_fuse_r_new = copy.deepcopy((1 - aep_fuse).T)
    del aep_fuse
    matchedpairs = {}
    for _ in range(2):
        aep_fuse_new, aep_fuse_r_new, matched, matched_true, dic_row, dic_col, matchedpairs \
            = ite1(aep_fuse_new, aep_fuse_r_new, dic_row, dic_col, matchedpairs, args.mode, d.test)
        total_matched += matched
        total_true += matched_true
        if matched == 0: break

    # for item in matchedpairs:
    #     ouf.write(str(item) + '\t' + str(matchedpairs[item]) + '\n')
    # print('Total Match ' + str(total_matched))
    # print('Total Match True ' + str(total_true))
    # print("End of pre-treatment...\n")

    new2old_row, new2old_col = dic_row, dic_col
    del dic_row
    del dic_col
    aep_fuse_new = 1 - aep_fuse_new
    leftids, rightids, newindex, cans, scores = evarerank(aep_fuse_new, new2old_row, new2old_col)

    leftids = np.array(leftids)
    rightids = np.array(rightids)

    if args.mode == "mul" or args.mode == "unm":
        M1 = np.zeros((len(d.test_lefts), len(d.test_lefts)))
        M2 = np.zeros((len(d.test_rights), len(d.test_rights)))
        Real2mindex_row = {d.test_lefts[i]: i for i in range(len(d.test_lefts))}  # test_lefts
        Real2mindex_col = {d.test_rights[i]: i for i in range(len(d.test_rights))}  # test_rights
        for item in d.KG1:
            if item[0] in d.test_lefts and item[2] in d.test_lefts:
                M1[Real2mindex_row[item[0]], Real2mindex_row[item[2]]] = 1
        for item in d.KG2:
            if item[0] in d.test_rights and item[2] in d.test_rights:
                M2[Real2mindex_col[item[0]], Real2mindex_col[item[2]]] = 1
    else:
        nelinenum = 10500
        M1 = np.zeros((len(d.test), len(d.test)))
        M2 = np.zeros((len(d.test), len(d.test)))
        for item in d.KG1:
            if item[0] < len(d.test) and item[2] < len(d.test):
                M1[item[0], item[2]] = 1
        for item in d.KG2:
            if item[0] - nelinenum < len(d.test) and item[2] - nelinenum < len(d.test):
                M2[item[0] - nelinenum, item[2] - nelinenum] = 1
    ### RL
    norm = 1
    OUTPUT_GRAPH = False
    GAMMA = 0.9  # reward discount in TD error
    LR_A = 0.001  # learning rate for actor
    LR_C = 0.002  # learning rate for critic
    truncNum = 10
    N_F = truncNum
    N_A = truncNum
    sess = tf.Session()
    actor = Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A)
    critic = Critic(sess, GAMMA, N_F,
                    lr=LR_C)  # we need a good teacher, so the teacher should learn faster than the actor
    sess.run(tf.global_variables_initializer())
    entNum = len(rightids)
    # t = time.time()
    epoch = 30  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    highest = 0
    RECORD = []
    for i_episode in range(epoch):
        # if i_episode % 30 == 0:
        #     print('epoch ' + str(i_episode))
        golScoreWhole = np.array([1.0] * entNum)
        trueacts = []
        ids = []
        idl2r = matchedpairs
        if args.mode == "mul" or args.mode == "unm":
            adj = np.where(M1[Real2mindex_row[leftids[newindex[0]]]] == 1)[0]
        else:
            adj = np.where(M1[leftids[newindex[0]]] == 1)[0]
        if len(adj) > 0:
            rids = []
            for id in adj:
                if id in idl2r:
                    if args.mode == "mul" or args.mode == "unm":
                        rids.append(Real2mindex_col[idl2r[id]])  # !!!!!!!!!!!!!!!!!!!!!!!!
                    else:
                        rids.append(idl2r[id])
            if args.mode == "mul" or args.mode == "unm":
                inds = []
                for item in cans[0]:
                    inds.append(Real2mindex_col[rightids[item]])
                Ms = M2[inds]
            else:
                Ms = M2[rightids[cans[0]]]
            Ms = Ms[:, rids]
            cohScore = np.sum(Ms, axis=-1).squeeze() / norm
        else:
            cohScore = np.array([0.0] * truncNum)

        golScore = golScoreWhole[cans[0]]
        locScore = scores[0]
        observation = locScore * golScore + cohScore
        for i in range(len(leftids)):
            action = actor.choose_action(observation)
            trueaction = cans[i][action]
            trueacts.append(trueaction)
            idl2r[leftids[newindex[i]]] = rightids[trueaction]
            ids.append(i)
            golScoreWhole[trueaction] = -1
            reward = (locScore * golScore + cohScore)[action]
            if i == len(leftids) - 1: break
            golScore_ = golScoreWhole[cans[i + 1]]
            locScore_ = scores[i + 1]
            if args.mode == "mul" or args.mode == "unm":
                adj = np.where(M1[Real2mindex_row[leftids[newindex[i + 1]]]] == 1)[0]
            else:
                adj = np.where(M1[leftids[newindex[i + 1]]] == 1)[0]
            if len(adj) > 0:
                rids = []
                for id in adj:
                    if id in idl2r:
                        if args.mode == "mul" or args.mode == "unm":
                            rids.append(Real2mindex_col[idl2r[id]])  ####!!!
                        else:
                            rids.append(idl2r[id])
                if args.mode == "mul" or args.mode == "unm":
                    inds = []
                    for item in cans[i + 1]:
                        inds.append(Real2mindex_col[rightids[item]])
                    Ms = M2[inds]
                else:
                    Ms = M2[rightids[cans[i + 1]]]
                Ms = Ms[:, rids]
                cohScore_ = np.sum(Ms, axis=-1).squeeze() / norm
            else:
                cohScore_ = np.array([0.0] * truncNum)
            observation_ = locScore_ * golScore_ + cohScore_
            td_error = critic.learn(observation, reward, observation_)  # gradient = grad[r + gamma * V(s_) - V(s)]
            actor.learn(observation, action, td_error)  # true_gradient = grad[logPi(s,a) * td_error]
            golScore = golScore_
            locScore = locScore_
            cohScore = cohScore_
            observation = observation_

        if args.mode == "mul" or args.mode == "unm":
            truth = 0
            for i in range(len(ids)):
                l = leftids[newindex[ids[i]]]
                r = rightids[trueacts[i]]
                if (l, r) in d.test:
                    truth += 1
            RECORD.append(truth)
        else:
            truth = np.where(rightids[trueacts] == leftids[newindex[ids].tolist()])
            # if i_episode == epoch-1:
            #     for iii in range(len(trueacts)):
            #         ouf.write(str(leftids[newindex[ids[iii]]]) + '\t' + str(rightids[trueacts[iii]]) + '\n')
            RECORD.append(len(truth[0]))
    RECORD = np.array(RECORD)
    trueC = total_true + np.average(RECORD[-20:])
    print("Total true is " +  str(trueC))

    if args.mode == "mul":
        precision = trueC * 1.0 / ent_num
        recall = trueC * 1.0 / len(d.test)
        f1 = 2 / (1 / precision + 1 / recall)
        print(str(precision) + "\t" + str(recall) + "\t" + str(f1))
    elif args.mode == "unm":
        precision = trueC * 1.0 / len(ids)
        recall = trueC * 1.0 / 10500
        f1 = 2 / (1 / precision + 1 / recall)
        print(str(precision) + "\t" + str(recall) + "\t" + str(f1))
    else:
        print("Acc: " + str(trueC) + ' / ' + str(ent_num) + ' = ' + str(trueC * 1.0 / ent_num))
