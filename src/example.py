import entmatcher as em
import entmatcher.modules.similarity as sim
import entmatcher.modules.score as score
import entmatcher.modules.matching as matching
from entmatcher.extras.data import Dataset
import numpy as np
import time

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="1-to-1", help="usage of features")  # 1-to-1 mul unm

    parser.add_argument("--encoder", type=str, default="gcn", help="the type of encoder")
    parser.add_argument("--features", type=str, default="stru", help="usage of features") # stru name struname
    parser.add_argument("--data_dir", type=str, default="../data/ja_en", required=False, help="input dataset file directory")  # mul | zh_en | ja_en | fr_en en_fr_15k_V1 en_de_15k_V1 dbp_wd_15k_V1 dbp_yg_15k_V1 fb_dbp
    parser.add_argument("--first_time", default='True', action='store_true', help='firsttime runing?')

    parser.add_argument("--sim", type=str, default="cosine",help="similarity metric")  # cosine euclidean manhattan
    parser.add_argument("--scoreop", type=str, default="none",help="score optimization strategy")  # csls sinkhorn rinf none
    parser.add_argument("--match", type=str, default="greedy", help="inference strategy") # hun sm rl greedy

    parser.add_argument("--multik", type=int, default=5, help="topk process of csls and recip")
    parser.add_argument("--sinkhornIte", type=int, default=100, help="iteration of sinkhorn")
    args = parser.parse_args()
    print(args)

    assert args.sim in ["cosine", "euclidean", "manhattan"]

    args.first_time = True

    # sim.getsim_matrix(se_vec, test_lefts, test_rights, method)
    d = Dataset(args)

    # choices of the representation learning model
    if args.encoder == "rrea":
        if args.first_time:
            se_vec = np.load(args.data_dir + '/vec-new.npy')
            # choices of the similarity matrix!!!
            aep = sim.get(se_vec, d.test_lefts, d.test_rights, args.sim)
            del se_vec
            np.save(args.data_dir + '/stru_mat_rrea.npy', aep)
        else:
            aep = np.load(args.data_dir + '/stru_mat_rrea.npy')
    elif args.encoder == "gcn":
        if args.first_time:
            se_vec = np.load(args.data_dir + '/vec.npy')
            # choices of the similarity matrix!!!
            aep = sim.get(se_vec, d.test_lefts, d.test_rights, args.sim)
            del se_vec
            np.save(args.data_dir + '/stru_mat_gcn.npy', aep)
        else:
            aep = np.load(args.data_dir + '/stru_mat_gcn.npy')
    else:
        print("FAlSE!!!")

    # choices of the input features
    if args.features == 'stru':
        aep_fuse = aep
        del aep
    elif args.features == 'name' or args.features == 'struname':
        if args.first_time:
            ne_vec = d.loadNe()
            aep_n = sim.get(ne_vec, d.test_lefts, d.test_rights, args.sim)
            np.save(args.data_dir + '/name_mat.npy', aep_n)
        else:
            aep_n = np.load(args.data_dir + '/name_mat.npy')

        if args.features == 'name':
            del aep
            aep_fuse = aep_n
            del aep_n
        else:
            def fusion(a, b):
                c = 0.5 * a + 0.5 * b
                return c
            aep_fuse = fusion(aep_n, aep)
            del aep_n
            del aep
    else:
        print("FAlSE!!!")

    aep_fuse = 1 - aep_fuse  # convert to similarity matrix
    t = time.time()
    aep_fuse = score.optimize(aep_fuse, args.scoreop, args)
    matching.matching(aep_fuse, d, args.match, args)
