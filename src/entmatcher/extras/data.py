import numpy as np


def loadfile(fn, num=1):
    # print('loading a file...' + fn)
    ret = []
    with open(fn, encoding='utf-8') as f:
        for line in f:
            th = line.split('\t')
            x = []
            for i in range(num):
                x.append(int(th[i]))
            ret.append(tuple(x))
    return ret

class Dataset:
    def __init__(self, args):
        self.e1 = args.data_dir + '/ent_ids_1'
        self.e2 = args.data_dir + '/ent_ids_2'
        self.ref = args.data_dir + '/ref_ent_ids'
        self.sup = args.data_dir + '/sup_ent_ids'
        self.val = args.data_dir + '/val_ent_ids'
        self.kg1 = args.data_dir + '/triples_1'
        self.kg2 = args.data_dir + '/triples_2'
        #optional
        self.nepath = args.data_dir + '/name_trans_vec_ftext.txt'

        ## for unmatchable scenario!!!
        if args.mode == "unm":
            self.e, self.r, self.KG1, self.KG2, self.train, self.test, self.test_lefts, self.test_rights, self.l2r, self.r2l, self.vali = \
                self.prepare_input_unm()
        else:
            self.e, self.r, self.KG1, self.KG2, self.train, self.test, self.test_lefts, self.test_rights, self.l2r, self.r2l, self.vali = \
                self.prepare_input()

    def prepare_input(self):
        e = len(set(loadfile(self.e1, 1)) | set(loadfile(self.e2, 1)))
        print(e)
        test = loadfile(self.ref, 2)
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
        train = loadfile(self.sup, 2)
        train = np.array(train)
        KG1 = loadfile(self.kg1, 3)
        KG2 = loadfile(self.kg2, 3)
        r = len(set([t1[1] for t1 in KG1]) | set([t2[1] for t2 in KG2]))
        vali = loadfile(self.val, 2)
        return e, r, KG1, KG2, train, test, test_lefts, test_rights, l2r, r2l, vali

    def prepare_input_unm(self):
        e = len(set(loadfile(self.e1, 1)) | set(loadfile(self.e2, 1)))
        print(e)
        test = loadfile(self.ref, 2)
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
        train = loadfile(self.sup, 2)
        train = np.array(train)
        KG1 = loadfile(self.kg1, 3)
        KG2 = loadfile(self.kg2, 3)
        r = len(set([t1[1] for t1 in KG1]) | set([t2[1] for t2 in KG2]))
        vali = loadfile(self.val, 2)
        # test_lefts also need to add the source entities!!!
        # add some entities into test_lefts!!!
        inf = open(self.e1)
        for i, line in enumerate(inf):
            strs = line.strip().split('\t')
            if i >= 15000:
                if int(strs[0]) not in test_lefts:
                    test_lefts.append(int(strs[0]))
        return e, r, KG1, KG2, train, test, test_lefts, test_rights, l2r, r2l, vali

    def loadNe(self):
        f1 = open(self.nepath)
        vectors = []
        for i, line in enumerate(f1):
            id, word, vect = line.rstrip().split('\t', 2)
            vect = np.fromstring(vect, sep=' ')
            vectors.append(vect)
        embeddings = np.vstack(vectors)
        return embeddings


