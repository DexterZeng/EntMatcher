import numpy as np
from scipy import spatial
import scipy
import tensorflow as tf

#新增:tf1.x-2.x
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.compat.v1.reset_default_graph()


def a():
    return "ddd"

def get(se_vec, test_lefts, test_rights, metric):
    if metric == "cosine":
        aep = cos(se_vec, test_lefts, test_rights)
    elif metric == "euclidean":
        aep = euc(se_vec, test_lefts, test_rights)
    elif metric == "manhattan":
        aep = manh(se_vec, test_lefts, test_rights)
    else:
        print("FAlSE!!!")
    return aep

# def cos(se_vec, test_lefts, test_rights):
#     Lvec = np.array([se_vec[e1] for e1 in test_lefts])
#     Rvec = np.array([se_vec[e2] for e2 in test_rights])
#     Lvec = Lvec / np.linalg.norm(Lvec, axis=-1, keepdims=True)
#     Rvec = Rvec / np.linalg.norm(Rvec, axis=-1, keepdims=True)
#     sim_mat = np.matmul(Lvec, Rvec.T)
#     aep = 1 - sim_mat
#     return aep

def cos(se_vec, test_lefts, test_rights):
    Lvec = tf.placeholder(tf.float32, [None, se_vec.shape[1]])
    Rvec = tf.placeholder(tf.float32, [None, se_vec.shape[1]])
    he = tf.nn.l2_normalize(Lvec, axis=-1)
    norm_e_em = tf.nn.l2_normalize(Rvec, axis=-1)
    aep = tf.matmul(he, tf.transpose(norm_e_em))
    sess = tf.Session()
    Lv = np.array([se_vec[e1] for e1 in test_lefts])
    Rv = np.array([se_vec[e2] for e2 in test_rights])
    aep = sess.run(aep, feed_dict={Lvec: Lv, Rvec: Rv})
    aep = 1-aep
    return aep

def euc(se_vec, test_lefts, test_rights):
    aep = getsim_matrix(se_vec, test_lefts, test_rights, "euclidean")
    return aep

def manh(se_vec, test_lefts, test_rights):
    aep = getsim_matrix(se_vec, test_lefts, test_rights, "cityblock")
    return aep

def getsim_matrix(se_vec, test_lefts, test_rights, method):
    Lvec = np.array([se_vec[e1] for e1 in test_lefts])
    Rvec = np.array([se_vec[e2] for e2 in test_rights])
    aep = scipy.spatial.distance.cdist(Lvec, Rvec, metric=method)
    return aep