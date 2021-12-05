from infer_helper import *
from scipy.optimize import linear_sum_assignment

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="1-to-1", help="usage of features")  # 1-to-1 mul unm

    parser.add_argument("--encoder", type=str, default="gcn", help="the type of encoder")
    parser.add_argument("--features", type=str, default="stru", help="usage of features") # stru name struname
    parser.add_argument("--data_dir", type=str, default="ja_en", required=False, help="input dataset file directory")  # 1hop zh_en
    parser.add_argument("--first_time", default='True', action='store_true', help='firsttime runing?')

    parser.add_argument("--sim", type=str, default="cosine",help="similarity metric")  # cosine euclidean manhattan
    parser.add_argument("--scoreop", type=str, default="none",help="score optimization strategy")  # csls sinkhorn rinf none
    parser.add_argument("--match", type=str, default="rl", help="inference strategy") # hun sm rl greedy

    parser.add_argument("--multik", type=int, default=5, help="topk process of csls and recip")
    parser.add_argument("--sinkhornIte", type=int, default=100, help="iteration of sinkhorn")
    args = parser.parse_args()
    print(args)

    args.first_time = True
    language = args.data_dir.split('/')[-1]  # mul | zh_en | ja_en | fr_en en_fr_15k_V1 en_de_15k_V1 dbp_wd_15k_V1 dbp_yg_15k_V1 fb_dbp
    e1 = 'data/' + language + '/ent_ids_1'
    e2 = 'data/' + language + '/ent_ids_2'
    ref = 'data/' + language + '/ref_ent_ids'
    sup = 'data/' + language + '/sup_ent_ids'
    val = 'data/' + language + '/val_ent_ids'
    kg1 = 'data/' + language + '/triples_1'
    kg2 = 'data/' + language + '/triples_2'

    ## for unmatchable scenario!!!
    if args.mode == "unm":
        e, KG1, KG2, train, test, test_lefts, test_rights, l2r, r2l, vali = prepare_input_unm(e1,e2,ref,sup,kg1,kg2,val)
    else:
        e, KG1, KG2, train, test, test_lefts, test_rights, l2r, r2l, vali = prepare_input(e1,e2,ref,sup,kg1,kg2,val)

    # choices of the representation learning model
    if args.encoder == "rrea":
        if args.first_time:
            se_vec = np.load('./data/' + language + '/vec-new.npy')
            # choices of the similarity matrix!!!
            if args.sim == "cosine":
                if args.data_dir in ["dbp_wd_100", "dbp_yg_100"]:
                    aep = getsim_matrix_large(se_vec, test_lefts, test_rights)
                else:
                    aep = getsim_matrix_cosine_sep(se_vec, test_lefts, test_rights)
            elif args.sim == "euclidean":
                aep = getsim_matrix(se_vec, test_lefts, test_rights, "euclidean")
            elif args.sim == "manhattan":
                aep = getsim_matrix(se_vec, test_lefts, test_rights, "cityblock")
            else:
                print("FAlSE!!!")
            del se_vec
            np.save('./data/' + language + '/stru_mat_rrea.npy', aep)
        else:
            aep = np.load('./data/' + language + '/stru_mat_rrea.npy')
    elif args.encoder == "gcn":
        if args.first_time:
            se_vec = np.load('./data/' + language + '/vec.npy')
            # choices of the similarity matrix!!!
            if args.sim == "cosine":
                if args.data_dir in ["dbp_wd_100", "dbp_yg_100"]:
                    aep = getsim_matrix_large(se_vec, test_lefts, test_rights)
                else:
                    aep = getsim_matrix_cosine_sep(se_vec, test_lefts, test_rights)
            elif args.sim == "euclidean":
                aep = getsim_matrix(se_vec, test_lefts, test_rights, "euclidean")
            elif args.sim == "manhattan":
                aep = getsim_matrix(se_vec, test_lefts, test_rights, "cityblock")
            else:
                print("FAlSE!!!")
            del se_vec
            np.save('./data/' + language + '/stru_mat_gcn.npy', aep)
        else:
            aep = np.load('./data/' + language + '/stru_mat_gcn.npy')
    else:
        print("FAlSE!!!")

    # choices of the input features
    if args.features == 'stru':
        aep_fuse = aep
        del aep
    elif args.features == 'name' or args.features == 'struname':
        if args.first_time:
            nepath = './data/' + language + '/name_trans_vec_ftext.txt'
            ne_vec = loadNe(nepath)
            if args.sim == "cosine":
                aep_n = getsim_matrix_cosine_sep(ne_vec, test_lefts, test_rights)
            elif args.sim == "euclidean":
                aep_n = getsim_matrix(ne_vec, test_lefts, test_rights, "euclidean")
            elif args.sim == "manhattan":
                aep_n = getsim_matrix(ne_vec, test_lefts, test_rights, "cityblock")
            else:
                print("FAlSE!!!")
            np.save('./data/' + language + '/name_mat.npy', aep_n)
        else:
            aep_n = np.load('./data/' + language + '/name_mat.npy')

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


    aep_fuse = 1-aep_fuse # convert to similarity matrix
    t = time.time()
    if args.scoreop == "csls":
        # after this, still the similarity matrix
        if args.mode == "mul":
            aep_fuse = csls_sim(aep_fuse, args.multik)
        else:
            aep_fuse = csls_sim(aep_fuse, 1)
        print("finish create CSLS matrix: {:.4f} s".format(time.time() - t))
    elif args.scoreop == "rinf":
        # the lower the better, dis!!!
        if args.mode == "mul":
            aep_fuse = 1- recip_mul(aep_fuse, args.multik, t)
        else:
            aep_fuse = 1- recip(aep_fuse, t)
    elif args.scoreop == "sinkhorn":
        if args.mode == "mul":
            dev = "cpu"
            aep_fuse = matrix_sinkhorn(torch.tensor(1-aep_fuse, device=dev), args.sinkhornIte).cpu().detach().numpy()
            print(aep_fuse)
        else:
            # input to sinkhorn is a distance matrix
            aep_fuse = matrix_sinkhorn(torch.tensor(1-aep_fuse, device="cpu"),100).cpu().detach().numpy() #cuda
            # output of sinkhorn is a similarity matrix
            print(aep_fuse)
    else:
        aep_fuse = aep_fuse

    # obtain the optimized scores... that is
    # then lets move on to the matching stage!!
    if args.match == "greedy":
        if args.mode == "mul":
            mul_max(1-aep_fuse, test_lefts, test_rights, test)
        elif args.mode == "unm":
            eva_unm(aep_fuse)
            eva_unm_tbnns(aep_fuse)
        else:
            eva(aep_fuse)
    elif args.match == "hun":
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
                if (test_lefts[row_ind[i]], test_rights[col_ind[i]]) in test:
                    trueC += 1
            print(trueC)
            precision = trueC * 1.0 / ent_num
            recall = trueC * 1.0 / len(test)
            f1 = 2 / (1 / precision + 1 / recall)
            print(str(precision) + "\t" + str(recall) + "\t" + str(f1))
        elif args.mode == "unm":
            for i in range(len(row_ind)):
                if row_ind[i]< 10500:
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
            print("Acc: " + str(trueC) + ' / ' + str(ent_num) + ' = ' + str(trueC * 1.0 / ent_num))
    elif args.match == "sm":
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
        print("Generate the preference scores time elapsed: {:.4f} s".format(time.time() - t))
        def getmale(pref):
            for i in range(scale):
                lis = (pref[i] + gapp).tolist()
                MALE_PREFS[i] = lis
            return MALE_PREFS
        MALE_PREFS = getmale(pref)
        del pref
        print("Forming the preference scores time 1 elapsed: {:.4f} s".format(time.time() - t))
        for i in range(scale_1):
            FEMALE_PREFS[i + gapp] = pref_col[:, i].tolist()
        del pref_col
        print("Forming the preference scores time 2 elapsed: {:.4f} s".format(time.time() - t))
        matches = gale_shapley(set(range(scale)), MALE_PREFS, FEMALE_PREFS)
        del MALE_PREFS
        del FEMALE_PREFS
        print("Deferred acceptance time elapsed: {:.4f} s".format(time.time() - t))

        trueC = 0
        if args.mode == "mul":
            for match in matches:
                if (test_lefts[int(match[0])], test_rights[int(match[1]) - gapp]) in test:
                    trueC += 1
            print(trueC)
            print(len(matches))
            print(len(test))
            precision = trueC * 1.0 / len(matches)
            recall = trueC * 1.0 / len(test)
            f1 = 2 / (1 / precision + 1 / recall)
            print(str(precision) + "\t" + str(recall) + "\t" + str(f1))
        else:
            for match in matches:
                if int(match[0]) + gapp == int(match[1]):
                    trueC += 1
            if args.mode == "unm":
                precision = trueC * 1.0 / len(matches)
            else:
                precision = trueC * 1.0 / 10500
            recall = trueC * 1.0 / 10500
            f1 = 2 / (1 / precision + 1 / recall)
            print(len(matches))
            print(trueC)
            print(str(precision) + "\t" + str(recall) + "\t" + str(f1))
    elif args.match == "rl":
        if args.mode == "mul" or args.mode == "unm":
            dic_row = {i: test_lefts[i] for i in range(len(test_lefts))}  # test_lefts
            dic_col = {i: test_rights[i] for i in range(len(test_rights))}  # test_rights
        else:
            dic_row = {i: i for i in range(len(test))}
            dic_col = {i: i for i in range(len(test))}

        total_matched = 0
        total_true = 0
        aep_fuse_new = copy.deepcopy(1 - aep_fuse)
        aep_fuse_r_new = copy.deepcopy((1 - aep_fuse).T)
        del aep_fuse
        matchedpairs = {}
        for _ in range(2):
            aep_fuse_new, aep_fuse_r_new, matched, matched_true, dic_row, dic_col, matchedpairs \
                = ite1(aep_fuse_new, aep_fuse_r_new, dic_row, dic_col, matchedpairs, args.mode, test)
            total_matched += matched
            total_true += matched_true
            if matched == 0: break

        print('Total Match ' + str(total_matched))
        print('Total Match True ' + str(total_true))
        print("End of pre-treatment...\n")

        new2old_row, new2old_col = dic_row, dic_col
        del dic_row
        del dic_col
        aep_fuse_new = 1 - aep_fuse_new
        leftids, rightids, newindex, cans, scores = evarerank(aep_fuse_new, new2old_row, new2old_col)

        leftids = np.array(leftids)
        rightids = np.array(rightids)

        if args.mode == "mul" or args.mode == "unm":
            M1 = np.zeros((len(test_lefts), len(test_lefts)))
            M2 = np.zeros((len(test_rights), len(test_rights)))
            Real2mindex_row = {test_lefts[i]: i for i in range(len(test_lefts))}  # test_lefts
            Real2mindex_col = {test_rights[i]: i for i in range(len(test_rights))}  # test_rights
            for item in KG1:
                if item[0] in test_lefts and item[2] in test_lefts:
                    M1[Real2mindex_row[item[0]], Real2mindex_row[item[2]]] = 1
            for item in KG2:
                if item[0] in test_rights and item[2] in test_rights:
                    M2[Real2mindex_col[item[0]], Real2mindex_col[item[2]]] = 1
        else:
            nelinenum = 10500
            M1 = np.zeros((len(test), len(test)))
            M2 = np.zeros((len(test), len(test)))
            for item in KG1:
                if item[0] < len(test) and item[2] < len(test):
                    M1[item[0], item[2]] = 1
            for item in KG2:
                if item[0] - nelinenum < len(test) and item[2] - nelinenum < len(test):
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
        t = time.time()
        epoch = 30 # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        fig_loss = np.zeros([epoch])
        fig_accuracy = np.zeros([epoch])
        highest = 0
        RECORD = []
        for i_episode in range(epoch):
            if i_episode % 30 == 0:
                print('epoch ' + str(i_episode))
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
                    if (l, r) in test:
                        truth += 1
                print(truth)
                RECORD.append(truth)
                if truth > highest:
                    highest = truth
                fig_accuracy[i_episode] = truth
            else:
                truth = np.where(rightids[trueacts] == leftids[newindex[ids].tolist()])
                RECORD.append(len(truth[0]))
                if len(truth[0]) > highest:
                    highest = len(truth[0])
                fig_accuracy[i_episode] = len(truth[0])
            print("time elapsed: {:.4f} s".format(time.time() - t))

        RECORD = np.array(RECORD)
        print('Averaged correct matches: ' + str(np.average(RECORD[-20:])))
    print("total time elapsed: {:.4f} s".format(time.time() - t))