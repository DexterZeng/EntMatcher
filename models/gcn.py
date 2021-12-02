import tensorflow as tf
import time
import json
import scipy
from scipy import spatial
import copy
from gcn_utils import *
seed = 12306
np.random.seed(seed)


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="mul", required=False, help="input dataset file directory")  # zh_en mul 1hop # en_fr_15k_V1
    args = parser.parse_args()
    print(args)
    language = args.data_dir
    e1 = '../data/' + language + '/ent_ids_1'
    e2 = '../data/' + language + '/ent_ids_2'
    ref = '../data/' + language + '/ref_ent_ids'
    sup = '../data/' + language + '/sup_ent_ids'
    val = '../data/' + language + '/val_ent_ids'
    kg1 = '../data/' + language + '/triples_1'
    kg2 = '../data/' + language + '/triples_2'
    epochs_se = 300
    epochs_ae = 600
    se_dim = 300
    ae_dim = 100
    act_func = tf.nn.relu
    gamma = 3 # margin based loss, 3.0
    k = 25  # number of negative samples for each positive one
    seed = 3  # 30% of seeds
    beta = 0.9  # weight of SE
    e, KG1, KG2, train, test, test_lefts, test_rights, l2r, r2l, vali = prepare_input(e1, e2, ref, sup, val, kg1, kg2)

    #build SE
    output_layer, loss,= build_SE(se_dim, act_func, gamma, k, e, train, KG1 + KG2)
    se_vec, J = training(output_layer, loss, 25, epochs_se, train, e, k)
    print('loss:', J)
    np.save('../data/' + language + '/vec.npy', se_vec)
    # aep = getsim_matrix_cosine_sep(se_vec, test_lefts, test_rights)
    # np.save('../data/' + language + '/stru_mat_train.npy', aep)