from torch import Tensor
import torch
import numpy as np
import time
import tensorflow as tf
from scipy import spatial
import scipy
import copy
from memory_profiler import profile
import sys
from pympler import asizeof

# @profile
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

# @profile
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
    # add some entitie into test_lefts!!!
    inf = open(e1)
    for i, line in enumerate(inf):
        strs = line.strip().split('\t')
        if i >= 15000:
            if int(strs[0]) not in test_lefts:
                test_lefts.append(int(strs[0]))
    return e, KG1, KG2, train, test, test_lefts, test_rights, l2r, r2l, vali

# @profile
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

def mul_max(dis, test_lefts, test_rights, test):
    # detected_pairs = []
    coun = 0
    for i in range(dis.shape[0]):
        rank = dis[i, :].argsort()
        chosen = rank[0]
        if (test_lefts[i], test_rights[chosen]) in test:
            coun += 1
    # print(dis)
    # print(dis.shape)
    print(len(test))
    print(coun)
    precision = coun*1.0/len(dis)
    recall = coun*1.0/len(test)
    f1 = 2/(1/precision + 1/recall)
    print(str(precision) + "\t" + str(recall)+ "\t" + str(f1))

def mul_thresholded_confi(dis, test_lefts, test_rights, test, gap=0.01):
    def less_than_gap(score, gap):
        indexes = np.where(score<=np.min(score)+gap)[0]
        return indexes
    # only get those the with confidence..
    # how to define high confidence?
    # with distance score lower than avergage minimum
    # print(dis)
    thres = np.average(np.min(dis, axis=-1))
    print(thres)
    sorted_mat = np.partition(dis, 3, axis=1)
    ave_gap = np.mean(sorted_mat[:,1]-sorted_mat[:,0])
    # print(ave_gap)
    gap = ave_gap

    predicts = []
    coun = 0
    for i in range(dis.shape[0]):
        min_score = np.min(dis[i, :])
        if min_score< thres:
            indexes = less_than_gap(dis[i, :], gap)
            if len(indexes)>1:
                # print(dis[i, :][indexes])
                if len(indexes)> 500:
                    continue
            for index in indexes:
                predicts.append((test_lefts[i], test_rights[index]))
                if (test_lefts[i], test_rights[index]) in test:
                    coun += 1
        else:
            rank = dis[i, :].argsort()
            chosen = rank[0]
            predicts.append((test_lefts[i], test_rights[chosen]))
            if (test_lefts[i], test_rights[chosen]) in test:
                coun += 1
    print("gap= " + str(gap))
    # print(dis)
    # print(dis.shape)
    print(len(test))
    print(len(predicts))
    print(coun)
    precision = coun * 1.0 / len(predicts)
    recall = coun * 1.0 / len(test)
    f1 = 2 / (1 / precision + 1 / recall)
    print(str(precision) + "\t" + str(recall) + "\t" + str(f1))

def mul_thresholded(dis, test_lefts, test_rights, test, gap=0.001):
    def less_than_gap(score, gap):
        indexes = np.where(score<=np.min(score)+gap)[0]
        return indexes
    predicts = []
    coun = 0
    for i in range(dis.shape[0]):
        indexes = less_than_gap(dis[i, :], gap)
        if len(indexes)>1:
            # print(dis[i, :][indexes])
            if len(indexes)> 500:
                continue
        for index in indexes:
            predicts.append((test_lefts[i], test_rights[index]))
            if (test_lefts[i], test_rights[index]) in test:
                coun += 1
    print("gap= " + str(gap))
    # print(dis)
    # print(dis.shape)
    print(len(test))
    print(len(predicts))
    print(coun)
    precision = coun * 1.0 / len(predicts)
    recall = coun * 1.0 / len(test)
    f1 = 2 / (1 / precision + 1 / recall)
    print(str(precision) + "\t" + str(recall) + "\t" + str(f1))

import copy
def mul_thresholded_debug(dis, test_lefts, test_rights, l2r, r2l,  test, thres):
    coun = 0
    for i in range(dis.shape[0]):
        score = copy.deepcopy(dis[i, :])
        score.sort()
        if len(l2r[test_lefts[i]])>1:
            # print(score)
            # get the indice
            ans = l2r[test_lefts[i]]
            positions = []
            for an in ans:
                pos = test_rights.index(an)
                positions.append(pos)
            ranks = []
            rank = dis[i, :].argsort()
            # print(sim[i, :])
            for pos in positions:
                rank_index = np.where(rank == pos)[0][0]
                ranks.append(rank_index)
            print(ranks)
            print(score[:len(ranks)])
            print()

def view2(x):
    if x.dim() == 2:
        return x
    return x.view(-1, x.size(-1))


def view3(x: Tensor) -> Tensor:
    if x.dim() == 3:
        return x
    return x.view(1, x.size(0), -1)

from fml.functional import sinkhorn

# @profile
def matrix_sinkhorn(pred_or_m, iter, expected=None, a=None, b=None):
    # print(asizeof.asizeof(pred_or_m))
    device = pred_or_m.device
    M = view3(pred_or_m).to(torch.float32)
    m, n = tuple(pred_or_m.size())
    a = torch.ones([1, m], requires_grad=False, device=device)
    b = torch.ones([1, n], requires_grad=False, device=device)
    P = sinkhorn(a, b, M, 1e-3, max_iters=iter, stop_thresh=1e-3) #max_iters=300
    return view2(P)

def get_hits(vec, test_pair, top_k=(1, 10, 50, 100)):
    Lvec = np.array([vec[e1] for e1, e2 in test_pair])
    Rvec = np.array([vec[e2] for e1, e2 in test_pair])
    sim = scipy.spatial.distance.cdist(Lvec, Rvec, metric='cosine')
    top_lr = [0] * len(top_k)
    mrr_sum_l = 0
    for i in range(Lvec.shape[0]):
        rank = sim[i, :].argsort()
        # print(sim[i, :])
        rank_index = np.where(rank == i)[0][0]
        mrr_sum_l = mrr_sum_l + 1.0 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1
    print('For each left:')
    for i in range(len(top_lr)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100))
    print("MRR: " + str(mrr_sum_l / len(test_pair)))

def get_hits_ma(sim, test_pair, top_k=(1, 10)):
    top_lr = [0] * len(top_k)
    mrr_sum_l = 0
    for i in range(sim.shape[0]):
        rank = sim[i, :].argsort()
        # print(sim[i, :])
        rank_index = np.where(rank == i)[0][0]
        mrr_sum_l = mrr_sum_l + 1.0 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1
    msg = 'Hits@1:%.3f, Hits@10:%.3f, MRR:%.3f\n' % (top_lr[0] / len(test_pair), top_lr[1] / len(test_pair), mrr_sum_l / len(test_pair))
    print(msg)

def ground_eva(sim):
    rank_sum = 0
    gap_sum = 0
    for i in range(sim.shape[0]):
        rank = sim[i, :].argsort()
        # print(sim[i, :])
        rank_index = np.where(rank == i)[0][0]
        rank_sum += rank_index
        sim_min = np.min(sim[i, :])
        sim_g = sim[i, :][i] #??????
        gap = sim_g-sim_min
        gap_sum+=gap

    print(rank_sum * 1.0 / sim.shape[0])
    print(gap_sum * 1.0 / sim.shape[0])


def ratio(sim):
    # ground truth ratio and the result ratio
    ground = np.trace(sim)
    maxx = np.sum(np.max(sim, axis=1))
    maxx_ = np.sum(np.max(sim, axis=0))
    per = ground * 1.0 / maxx
    per_ = ground * 1.0 / maxx_
    print(per)
    print(per_)
    print((per+per_)/2)

    # this ratio characterzies the sum of ground truth divided by the sum of the direct align
    # (always align to the most similar one)

def topk(sim, top_k=(2, 5, 10)):
    top_lr = [0] * len(top_k)
    for i in range(sim.shape[0]):
        rank = sim[i, :].argsort()
        # print(sim[i, :])
        rank_index = np.where(rank == i)[0][0]

        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1
    msg = 'Hits@2:%.3f, Hits@5:%.3f, Hits@10:%.3f\n' % (top_lr[0] / len(sim), top_lr[1] / len(sim), top_lr[2] / len(sim))
    print(msg)

def cnt_many2one(sim):
    rights = set()
    rights_mul = set()
    cnt = 0
    for i in range(sim.shape[0]):
        rank = sim[i, :].argsort()
        if rank[0] in rights:
            cnt += 1
            if rank[0] not in rights_mul:
                cnt += 1
                rights_mul.add(rank[0])
        rights.add(rank[0])
    per1 = cnt*1.0/sim.shape[0]
    print("mul2one percentage : " + str(per1))

    lefts = set()
    lefts_mul = set()
    cnt = 0
    for i in range(sim.shape[1]):
        rank = sim[:, i].argsort()
        if rank[0] in lefts:
            cnt += 1
            if rank[0] not in lefts_mul:
                cnt += 1
                lefts_mul.add(rank[0])
        lefts.add(rank[0])
    per2 = cnt * 1.0 / sim.shape[1]
    print("mul2one percentage : " + str(per2))

    print("mul2one percentage ave : " + str((per1 + per2)/2))


def get_hits_ma_bi(sim, test_pair, top_k=(1, 10)):
    # get the bidirectional results
    top_lr = [0] * len(top_k)
    mrr_sum_l = 0
    for i in range(sim.shape[0]):
        rank = sim[i, :].argsort()
        # print(sim[i, :])
        rank_index = np.where(rank == i)[0][0]
        mrr_sum_l = mrr_sum_l + 1.0 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1
    msg = 'LEFT Hits@1:%.3f, Hits@10:%.3f, MRR:%.3f\n' % (top_lr[0] / len(test_pair), top_lr[1] / len(test_pair), mrr_sum_l / len(test_pair))
    print(msg)

    top_lr = [0] * len(top_k)
    mrr_sum_l = 0
    for i in range(sim.shape[1]):
        rank = sim[:, i].argsort()
        # print(sim[i, :])
        rank_index = np.where(rank == i)[0][0]
        mrr_sum_l = mrr_sum_l + 1.0 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1
    msg = 'RIGHT Hits@1:%.3f, Hits@10:%.3f, MRR:%.3f\n' % (top_lr[0] / len(test_pair), top_lr[1] / len(test_pair), mrr_sum_l / len(test_pair))
    print(msg)


def get_hits_ma_bi_agg(sim, test_pair, top_k=(1, 10)):
    # get the bidirectional results, and aggregate to reach a aggrement?
    # I guess this is improved stable matching, suitable in our setting, mMMS比较多, do not have to be 1-to-1 constrained!!!
    top_lr = [0] * len(top_k)
    mrr_sum_l = 0
    left2right = dict()
    for i in range(sim.shape[0]):
        rank = sim[i, :].argsort()
        left2right[i] = rank[0]
        # print(sim[i, :])
        rank_index = np.where(rank == i)[0][0]
        mrr_sum_l = mrr_sum_l + 1.0 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1
    msg = 'LEFT Hits@1:%.3f, Hits@10:%.3f, MRR:%.3f' % (top_lr[0] / len(test_pair), top_lr[1] / len(test_pair), mrr_sum_l / len(test_pair))
    print(msg)

    top_lr = [0] * len(top_k)
    mrr_sum_l = 0
    right2left = dict()
    for i in range(sim.shape[1]):
        rank = sim[:, i].argsort()
        right2left[i] = rank[0]
        # print(sim[i, :])
        rank_index = np.where(rank == i)[0][0]
        mrr_sum_l = mrr_sum_l + 1.0 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1
    msg = 'RIGHT Hits@1:%.3f, Hits@10:%.3f, MRR:%.3f' % (top_lr[0] / len(test_pair), top_lr[1] / len(test_pair), mrr_sum_l / len(test_pair))
    print(msg)

    # merge the results by keeping the mutual NNS
    mNNS = []
    cor = 0
    rest_left = []
    chosen_right = set()
    for i in range(len(left2right)):
        if right2left[left2right[i]] == i:
            mNNS.append([i, left2right[i]])
            chosen_right.add(left2right[i])
            if i == left2right[i]:
                cor +=1
        else:
            rest_left.append(i)
    # rest_right = list(set(range(len(right2left))).difference(set(chosen_right))) to keep the order, cannot use this
    rest_right = []
    for i in range(len(right2left)):
        if i not in chosen_right:
            rest_right.append(i)

    assert len(rest_left) == len(rest_right)
    print("First time acc: " + str(cor) + '/' + str(len(mNNS))+ ' = '+ str(cor * 1.0 / len(mNNS)))

    while len(rest_left)>0:
        rest_sim = sim[rest_left,:][:, rest_right]
        left2right = dict()
        for i in range(rest_sim.shape[0]):
            rank = rest_sim[i, :].argsort()
            left2right[i] = rank[0]

        right2left = dict()
        for i in range(rest_sim.shape[0]):
            rank = rest_sim[:, i].argsort()
            right2left[i] = rank[0]

        rest_left_ = []
        chosen_right = set()
        for i in range(len(left2right)):
            if right2left[left2right[i]] == i:
                mNNS.append([rest_left[i], rest_right[left2right[i]]])
                chosen_right.add(rest_right[left2right[i]])
            else:
                rest_left_.append(rest_left[i])

        rest_right_ = []
        for i in rest_right:
            if i not in chosen_right:
                rest_right_.append(i)

        rest_left = copy.deepcopy(rest_left_)
        rest_right = copy.deepcopy(rest_right_)
        assert len(rest_left) == len(rest_right)

    cor = 0
    for item in mNNS:
        if item[0] == item[1]:
            cor += 1
    print("Final acc: " + str(cor) + '/' + str(len(mNNS))+ ' = '+ str(cor * 1.0 / len(mNNS)))

def get_size(obj, seen=None):
    # From
    # Recursively finds size of objects
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

# reciprocal inference
from scipy.stats import rankdata
# @profile
def recip(string_mat, t = None):
    # print(string_mat.dtype)
    max_value = np.max(string_mat, axis=0)
    # print(max_value.shape)
    max_value[max_value == 0.0] = 1.0
    a = string_mat - max_value + 1
    a = -a
    a_rank = rankdata(a, axis=1)
    # print(asizeof.asizeof(a_rank))
    # print(a_rank)
    # print(a_rank.dtype)
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

def male_without_match(matches, males):
    for male in males:
        if male not in matches:
            return male

from collections import defaultdict

# @profile
def deferred_acceptance(male_prefs, female_prefs):
    female_queue = defaultdict(int)
    males = list(male_prefs.keys())
    matches = {}
    while True:
        male = male_without_match(matches, males)
        # print(male)
        if male is None:
            break
        female_index = female_queue[male]
        female_queue[male] += 1
        # print(female_index)
        try:
            female = male_prefs[male][female_index]
        except IndexError:
            matches[male] = male
            continue
        # print('Trying %s with %s... ' % (male, female), end='')
        prev_male = matches.get(female, None)
        if not prev_male:
            matches[male] = female
            matches[female] = male
            # print('auto')
        elif female_prefs[female].index(male) < \
             female_prefs[female].index(prev_male):
            matches[male] = female
            matches[female] = male
            del matches[prev_male]
    return {male: matches[male] for male in male_prefs.keys()}


from collections import deque

def pref_to_rank(pref):
    return {
        a: {b: idx for idx, b in enumerate(a_pref)}
        for a, a_pref in pref.items()
    }

# @profile
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
    #
    return [(a, b) for b, a in pair.items()]

# @profile
def calculate_nearest_k(sim_mat, k):
    sorted_mat = -np.partition(-sim_mat, k + 1, axis=1)  # -np.sort(-sim_mat1)
    nearest_k = sorted_mat[:, 0:k]
    return np.mean(nearest_k, axis=1, keepdims=True)

# @profile
def csls_sim(sim_mat, k):
    nearest_values1 = calculate_nearest_k(sim_mat, k)
    nearest_values2 = calculate_nearest_k(sim_mat.T, k)
    csls_sim_mat = 2 * sim_mat - nearest_values1 - nearest_values2.T
    del sim_mat
    return csls_sim_mat


# load dataset
# load a file and return a list of tuple containing $num integers in each line
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
    #print(confident)
    print('Confi in fuse: ' + str(len(confident)))
    correct = 0
    if mode == 'mul' or mode == 'unm':
        for i in confident:
            # print(dic_row[i])
            # print(confident[i])
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
    #print(len(new2old_row))
    #print(len(new2old_col))
    aep_fuse_new = np.delete(aep_fuse, row, axis=0)
    aep_fuse_new = np.delete(aep_fuse_new, col, axis=1)
    aep_fuse_r_new = aep_fuse_new.T
    # M1 = np.delete(M1, row, axis=0)
    # M1 = np.delete(M1, row, axis=1)
    # M2 = np.delete(M2, col, axis=0)
    # M2 = np.delete(M2, col, axis=1)
    return aep_fuse_new, aep_fuse_r_new, len(confident), correct, new2old_row, new2old_col, mmtached

def evarerank_mul(aep_fuse_new, new2old_row, new2old_col, test):
    ### reindex according to difficulty
    aep_fuse = aep_fuse_new
    print(aep_fuse.shape)
    aep_rank = copy.deepcopy(aep_fuse)
    aep_rank = -aep_rank

    results = aep_rank.argsort(axis=1)[:, 0]
    correct = 0
    leftids = []
    pairs = []
    for i in range(len(new2old_row)):
        leftid = new2old_row[i]
        leftids.append(leftid)
        rightid = new2old_col[results[i]]
        if (leftid, rightid) in test:
            correct += 1
        pairs.append([leftid, rightid])

    print("maximum strategy " + str(correct))
    print()

    rightids = []
    for i in range(len(new2old_col)):
        rightid = new2old_col[i]
        rightids.append(rightid)

    # count = 0
    # for left in leftids:
    #     if left in rightids:
    #         count += 1
    # print("total " + str(count))

    top_ids = aep_rank.argsort(axis=1)[:, :10]  # top_ids seem to be the positions, instead of the actual ids...

    aep_rank.sort(axis=1)
    top_scores = -aep_rank[:, :10]
    scores = -aep_rank[:, 0]
    # truescores = aep_fuse[wrongsx, wrongsy]
    # gap = scores[wrongsx] - truescores
    # print(len(gap))
    # print(np.mean(gap))

    # print(scores)
    newindex = (-scores).argsort()
    # print(newindex)
    cans = top_ids[newindex, :]
    scores = top_scores[newindex, :]
    # print(cans)
    # print(scores)

    return leftids, rightids, newindex, cans, scores


def evarerank(aep_fuse_new, new2old_row, new2old_col):
    ### reindex according to difficulty
    aep_fuse = aep_fuse_new
    del aep_fuse_new
    # print(aep_fuse)
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
    # check multiple
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

    top_ids = aep_rank.argsort(axis=1)[:, :10]  # top_ids seem to be the positions, instead of the actual ids...
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

    #print(truePositions)
    print("top10 " + str(top10))

    aep_rank.sort(axis=1)
    top_scores = -aep_rank[:, :10]
    scores = -aep_rank[:, 0]
    # truescores = aep_fuse[wrongsx, wrongsy]
    # gap = scores[wrongsx] - truescores
    # print(len(gap))
    # print(np.mean(gap))

    # print(scores)
    newindex = (-scores).argsort()
    # print(newindex)
    cans = top_ids[newindex, :]
    scores = top_scores[newindex, :]
    # print(cans)
    # print(scores)

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
