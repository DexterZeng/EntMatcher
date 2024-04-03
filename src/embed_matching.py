import entmatcher as em
from entmatcher.modules import enhance
import numpy as np
import time
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="1-to-1", help="mode")  # 1-to-1 mul unm
    parser.add_argument("--encoder", type=str, default="rrea", help="the type of encoder")
    parser.add_argument("--features", type=str, default="name", help="usage of features") # stru name struname
    parser.add_argument("--enhance", type=str, default="none", required=False, help="enhancement method for representation") # tfp
    parser.add_argument("--data_dir", type=str, default="../data/zh_en", required=False, help="input dataset file directory")  # mul | zh_en | ja_en | fr_en en_fr_15k_V1 en_de_15k_V1 dbp_wd_15k_V1 dbp_yg_15k_V1 fb_dbp

    parser.add_argument("--algorithm", type=str, default="dinf", help="choice of algorithm")  # 1-to-1 mul unm

    parser.add_argument("--multik", type=int, default=5, help="topk process of csls and recip")
    parser.add_argument("--sinkhornIte", type=int, default=100, help="iteration of sinkhorn")
    args = parser.parse_args()
    print(args)

    t = time.time()
    # generate datasets
    d = em.extras.Dataset(args)
    # t = time.time()
    # choose one embedding matching algorithm
    if args.algorithm == "dinf":
        a1 = em.algorithms.DInf(args)
    elif args.algorithm == "csls":
        a1 = em.algorithms.CSLS(args)
    elif args.algorithm == "rinf":
        a1 = em.algorithms.RInf(args)
    elif args.algorithm == "sinkhorn":
        a1 = em.algorithms.Sinkhorn(args)
    elif args.algorithm == "hun":
        a1 = em.algorithms.Hun(args)
    elif args.algorithm == "sm":
        a1 = em.algorithms.SMat(args)
    elif args.algorithm == "rl":
        a1 = em.algorithms.RL(args)

    # choices of the representation learning model
    if args.encoder == "rrea":
        se_vec = np.load(args.data_dir + '/vec-new.npy')
    elif args.encoder == "gcn":
        se_vec = np.load(args.data_dir + '/vec.npy')
    else:
        print("FAlSE!!!")

    if args.enhance == "tfp":
        se_vec = enhance.gen(se_vec, d, "tfp")
    
    if args.features == 'stru':
        a1.match([se_vec], d)
    if args.features == 'name' or args.features == 'struname':
        ne_vec = d.loadNe()
        if args.features == 'name':
            a1.match([ne_vec], d)
        else:
            a1.match([se_vec, ne_vec], d)
    print("total time elapsed: {:.4f} s".format(time.time() - t))

