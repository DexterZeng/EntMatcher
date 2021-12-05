from torch import Tensor
import torch
import numpy as np
import time
import tensorflow as tf
from scipy import spatial
import scipy
import copy

def eva(sim_mat, use_min = False):
    if use_min is True:
        predicted = np.argmin(sim_mat, axis=1)
    else:
        predicted = np.argmax(sim_mat, axis=1)
    cor = predicted == np.array(range(sim_mat.shape[0]))
    cor_num = np.sum(cor)
    print("Acc: " + str(cor_num) + ' / ' + str(len(cor)) + ' = ' + str(cor_num*1.0/len(cor)))

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

def prepare_input(e1,e2,ref,sup,kg1,kg2,val):
    e = len(set(loadfile(e1, 1)) | set(loadfile(e2, 1)))
    print(e)
    test = loadfile(ref, 2)
    test_lefts = []
    test_rights = []
    l2r = dict()
    r2l = dict()
    for item in test:
        if item[0] not in test_lefts:
            test_lefts.append(item[0])
        if item[1] not in test_rights:
            test_rights.append(item[1])
        if item[0] not in l2r:
            ents = []
        else:
            ents = l2r[item[0]]
        ents.append(item[1])
        l2r[item[0]] = ents
        if item[1] not in r2l:
            ents = []
        else:
            ents = r2l[item[1]]
        ents.append(item[0])
        r2l[item[1]] = ents
    train = loadfile(sup, 2)
    train = np.array(train)
    KG1 = loadfile(kg1, 3)
    KG2 = loadfile(kg2, 3)
    vali = loadfile(val, 2)
    return e, KG1, KG2, train, test, test_lefts, test_rights, l2r, r2l, vali

def prepare_input_unm(e1,e2,ref,sup,kg1,kg2,val):
    e = len(set(loadfile(e1, 1)) | set(loadfile(e2, 1)))
    print(e)
    test = loadfile(ref, 2)
    # to keep the order, you cannot use set!!!
    test_lefts = list()
    test_rights = list()
    l2r = dict()
    r2l = dict()
    for item in test:
        if item[0] not in test_lefts:
            test_lefts.append(item[0])
        if item[1] not in test_rights:
            test_rights.append(item[1])
        if item[0] not in l2r:
            ents = []
        else:
            ents = l2r[item[0]]
        ents.append(item[1])
        l2r[item[0]] = ents
        if item[1] not in r2l:
            ents = []
        else:
            ents = r2l[item[1]]
        ents.append(item[0])
        r2l[item[1]] = ents
    train = loadfile(sup, 2)
    train = np.array(train)
    KG1 = loadfile(kg1, 3)
    KG2 = loadfile(kg2, 3)
    vali = loadfile(val, 2)
    # test_lefts also need to add the source entities!!!
    # add some entities into test_lefts!!!
    inf = open(e1)
    for i, line in enumerate(inf):
        strs = line.strip().split('\t')
        if i >= 15000:
            if int(strs[0]) not in test_lefts:
                test_lefts.append(int(strs[0]))
    return e, KG1, KG2, train, test, test_lefts, test_rights, l2r, r2l, vali

def getsim_matrix_large(se_vec, test_lefts, test_rights):
    Lvec = np.array([se_vec[e1] for e1 in test_lefts])
    Rvec = np.array([se_vec[e2] for e2 in test_rights])
    Lvec = Lvec / np.linalg.norm(Lvec, axis=-1, keepdims=True)
    Rvec = Rvec / np.linalg.norm(Rvec, axis=-1, keepdims=True)
    sim_mat = np.matmul(Lvec, Rvec.T)
    aep = 1 - sim_mat
    return aep

def getsim_matrix_cosine_sep(se_vec, test_lefts, test_rights):
    Lvec = tf.placeholder(tf.float32, [None, se_vec.shape[1]])
    Rvec = tf.placeholder(tf.float32, [None, se_vec.shape[1]])
    he = tf.nn.l2_normalize(Lvec, dim=-1)
    norm_e_em = tf.nn.l2_normalize(Rvec, dim=-1)
    aep = tf.matmul(he, tf.transpose(norm_e_em))
    sess = tf.Session()
    Lv = np.array([se_vec[e1] for e1 in test_lefts])
    Rv = np.array([se_vec[e2] for e2 in test_rights])
    aep = sess.run(aep, feed_dict={Lvec: Lv, Rvec: Rv})
    aep = 1-aep
    return aep

def getsim_matrix(se_vec, test_lefts, test_rights, method):
    Lvec = np.array([se_vec[e1] for e1 in test_lefts])
    Rvec = np.array([se_vec[e2] for e2 in test_rights])
    aep = scipy.spatial.distance.cdist(Lvec, Rvec, metric=method)
    return aep

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

def view2(x):
    if x.dim() == 2:
        return x
    return x.view(-1, x.size(-1))

def view3(x: Tensor) -> Tensor:
    if x.dim() == 3:
        return x
    return x.view(1, x.size(0), -1)

from fml.functional import sinkhorn

def matrix_sinkhorn(pred_or_m, iter, expected=None, a=None, b=None):
    device = pred_or_m.device
    M = view3(pred_or_m).to(torch.float32)
    m, n = tuple(pred_or_m.size())
    a = torch.ones([1, m], requires_grad=False, device=device)
    b = torch.ones([1, n], requires_grad=False, device=device)
    P = sinkhorn(a, b, M, 1e-3, max_iters=iter, stop_thresh=1e-3) #max_iters=300
    return view2(P)

# reciprocal inference
from scipy.stats import rankdata
def recip(string_mat, t = None):
    max_value = np.max(string_mat, axis=0)
    max_value[max_value == 0.0] = 1.0
    a = string_mat - max_value + 1
    a = -a
    a_rank = rankdata(a, axis=1)
    del a
    if t is not None:
        print("rank matrix 1 time elapsed: {:.4f} s".format(time.time() - t))
    max_value = np.max(string_mat, axis=1)
    max_value[max_value == 0.0] = 1.0
    b = (string_mat.T - max_value) + 1
    del string_mat
    del max_value
    b = -b
    b_rank = rankdata(b, axis=1)
    del b
    if t is not None:
        print("rank matrix 2 time elapsed: {:.4f} s".format(time.time() - t))
    b_rank = b_rank.T
    recip_sim = (a_rank + b_rank) / 2.0
    del a_rank
    del b_rank
    return recip_sim

# a multi version version of recip
def recip_mul(string_mat, k=1, t = None):
    sorted_mat = -np.partition(-string_mat, k + 1, axis=0)
    max_values = np.mean(sorted_mat[0:k, :], axis=0)
    a = string_mat - max_values + 1
    sorted_mat = -np.partition(-string_mat, k + 1, axis=1)
    max_values = np.mean(sorted_mat[:, 0:k], axis=1)
    b = (string_mat.T - max_values) + 1
    del string_mat
    del max_values
    if t is not None:
        print("Generate the preference matrix time elapsed: {:.4f} s".format(time.time() - t))
    from scipy.stats import rankdata
    a_rank = rankdata(-a, axis=1)
    del a
    b_rank = rankdata(-b, axis=1)
    del b
    if t is not None:
        print("Generate the ranking matrix time elapsed: {:.4f} s".format(time.time() - t))
    recip_sim = (a_rank + b_rank.T) / 2.0
    del a_rank
    del b_rank
    return recip_sim

from collections import deque
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

def calculate_nearest_k(sim_mat, k):
    sorted_mat = -np.partition(-sim_mat, k + 1, axis=1)  # -np.sort(-sim_mat1)
    nearest_k = sorted_mat[:, 0:k]
    return np.mean(nearest_k, axis=1, keepdims=True)

def csls_sim(sim_mat, k):
    nearest_values1 = calculate_nearest_k(sim_mat, k)
    nearest_values2 = calculate_nearest_k(sim_mat.T, k)
    csls_sim_mat = 2 * sim_mat - nearest_values1 - nearest_values2.T
    del sim_mat
    return csls_sim_mat

def loadfile(fn, num=1):
    print('loading a file...' + fn)
    ret = []
    with open(fn, encoding='utf-8') as f:
        for line in f:
            th = line[:-1].split('\t')
            x = []
            for i in range(num):
                x.append(int(th[i]))
            ret.append(tuple(x))
    return ret

def loadNe(path):
	f1 = open(path)
	vectors = []
	for i, line in enumerate(f1):
		id, word, vect = line.rstrip().split('\t', 2)
		vect = np.fromstring(vect, sep=' ')
		vectors.append(vect)
	embeddings = np.vstack(vectors)
	return embeddings

# rl
def ite1(aep_fuse, aep_fuse_r, dic_row, dic_col, mmtached, mode, test):
    print(aep_fuse.shape)
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
    print('Confi in fuse: ' + str(len(confident)))
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
    print('Correct in fuse: ' + str(correct))
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
    print(aep_fuse.shape)
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
    print("maximum strategy " + str(correct))
    print()
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
    print("multiple source entities are aligned to the same target entities!")
    print('source ' + str(mulsource))
    print('target ' + str(len(multiple)))
    print()
    rightids = []
    for i in range(len(new2old_col)):
        rightid = new2old_col[i]
        rightids.append(rightid)
    count = 0
    for left in leftids:
        if left in rightids:
            count += 1
    print("total " + str(count))
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
    print("top10 " + str(top10))
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