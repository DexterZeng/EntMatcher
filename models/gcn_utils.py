import tensorflow as tf
import math
import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.compat.v1.reset_default_graph()
'''
Adapted from https://github.com/tkipf/gcn
'''


def uniform(shape, scale=0.05, name=None):
	"""Uniform init."""
	initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
	return tf.Variable(initial, name=name)


def glorot(shape, name=None):
	"""Glorot & Bengio (AISTATS 2010) init."""
	init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
	initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
	return tf.Variable(initial, name=name)


def zeros(shape, name=None):
	"""All zeros."""
	initial = tf.zeros(shape, dtype=tf.float32)
	return tf.Variable(initial, name=name)


def ones(shape, name=None):
	"""All ones."""
	initial = tf.ones(shape, dtype=tf.float32)
	return tf.Variable(initial, name=name)

def prepare_input(e1, e2, ref, sup, val, kg1, kg2):
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

def func(KG):
	head = {}
	cnt = {}
	for tri in KG:
		if tri[1] not in cnt:
			cnt[tri[1]] = 1
			head[tri[1]] = set([tri[0]])
		else:
			cnt[tri[1]] += 1
			head[tri[1]].add(tri[0])
	r2f = {}
	for r in cnt:
		r2f[r] = len(head[r]) / cnt[r]
	return r2f


def ifunc(KG):
	tail = {}
	cnt = {}
	for tri in KG:
		if tri[1] not in cnt:
			cnt[tri[1]] = 1
			tail[tri[1]] = set([tri[2]])
		else:
			cnt[tri[1]] += 1
			tail[tri[1]].add(tri[2])
	r2if = {}
	for r in cnt:
		r2if[r] = len(tail[r]) / cnt[r]
	return r2if


def get_mat(e, KG):
	r2f = func(KG)
	r2if = ifunc(KG)
	du = [1] * e
	for tri in KG:
		if tri[0] != tri[2]:
			du[tri[0]] += 1
			du[tri[2]] += 1
	M = {}
	for tri in KG:
		# if tri[0] == tri[2]:
		# 	continue
		# if (tri[0], tri[2]) not in M:
		# 	M[(tri[0], tri[2])] = math.sqrt(math.sqrt(r2if[tri[1]]))
		# else:
		# 	M[(tri[0], tri[2])] += math.sqrt(math.sqrt(r2if[tri[1]]))
		# if (tri[2], tri[0]) not in M:
		# 	M[(tri[2], tri[0])] = math.sqrt(math.sqrt(r2f[tri[1]]))
		# else:
		# 	M[(tri[2], tri[0])] += math.sqrt(math.sqrt(r2f[tri[1]]))
		if (tri[0], tri[2]) not in M:
			M[(tri[0], tri[2])] = 1
		else:
			M[(tri[0], tri[2])] += 1
		if (tri[2], tri[0]) not in M:
			M[(tri[2], tri[0])] = 1
		else:
			M[(tri[2], tri[0])] += 1
	for i in range(e):
		M[(i, i)] = 1
	return M, du


# get a sparse tensor based on relational triples
def get_sparse_tensor(e, KG):
	print('getting a sparse tensor...')
	M, du = get_mat(e, KG)
	ind = []
	val = []
	for fir, sec in M:
		ind.append((sec, fir))
		val.append(M[(fir, sec)] / math.sqrt(du[fir]) / math.sqrt(du[sec]))
		# val.append(float(M[(fir, sec)]))
	M = tf.SparseTensor(indices=ind, values=val, dense_shape=[e, e])
	return M


# add a layer
def add_diag_layer(inlayer, dimension, M, act_func, dropout=0.0, init=ones):
	inlayer = tf.nn.dropout(inlayer, 1 - dropout)
	print('adding a layer...')
	w0 = init([1, dimension])
	tosum = tf.sparse_tensor_dense_matmul(M, tf.multiply(inlayer, w0))
	if act_func is None:
		return tosum
	else:
		return act_func(tosum)


def add_full_layer(inlayer, dimension_in, dimension_out, M, act_func, dropout=0.0, init=glorot):
	inlayer = tf.nn.dropout(inlayer, 1 - dropout)
	print('adding a layer...')
	w0 = init([dimension_in, dimension_out])
	tosum = tf.sparse_tensor_dense_matmul(M, tf.matmul(inlayer, w0))
	if act_func is None:
		return tosum
	else:
		return act_func(tosum)


# se input layer
def get_se_input_layer(e, dimension):
	print('adding the se input layer...')
	ent_embeddings = tf.Variable(tf.truncated_normal([e, dimension], stddev=1.0 / math.sqrt(e)))
	return tf.nn.l2_normalize(ent_embeddings, 1)


# ae input layer
def get_ae_input_layer(attr):
	print('adding the ae input layer...')
	return tf.constant(attr)


# get loss node
def get_loss(outlayer, ILL, gamma, k):
	print('getting loss...')
	left = ILL[:, 0]
	right = ILL[:, 1]
	t = len(ILL)
	left_x = tf.nn.embedding_lookup(outlayer, left)
	right_x = tf.nn.embedding_lookup(outlayer, right)
	A = tf.reduce_sum(tf.abs(left_x - right_x), 1)
	neg_left = tf.placeholder(tf.int32, [t * k], "neg_left")
	neg_right = tf.placeholder(tf.int32, [t * k], "neg_right")
	neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
	neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)
	B = tf.reduce_sum(tf.abs(neg_l_x - neg_r_x), 1)
	C = - tf.reshape(B, [t, k])
	D = A + gamma
	L1 = tf.nn.relu(tf.add(C, tf.reshape(D, [t, 1])))

	# A1 = tf.sqrt(tf.reduce_sum(tf.square(left_x - right_x), 1))
	# B1 = tf.sqrt(tf.reduce_sum(tf.square(neg_l_x - neg_r_x), 1))
	# C1 = - tf.reshape(B1, [t, k])
	# D1 = A1 + gamma
	# L11 = tf.nn.relu(tf.add(C1, tf.reshape(D1, [t, 1])))

	neg_left = tf.placeholder(tf.int32, [t * k], "neg2_left")
	neg_right = tf.placeholder(tf.int32, [t * k], "neg2_right")
	neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
	neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)
	B = tf.reduce_sum(tf.abs(neg_l_x - neg_r_x), 1)
	C = - tf.reshape(B, [t, k])
	L2 = tf.nn.relu(tf.add(C, tf.reshape(D, [t, 1])))

	# B1 = tf.sqrt(tf.reduce_sum(tf.square(neg_l_x - neg_r_x), 1))
	# C1 = - tf.reshape(B1, [t, k])
	# L21 = tf.nn.relu(tf.add(C1, tf.reshape(D1, [t, 1])))

	return (tf.reduce_sum(L1) + tf.reduce_sum(L2)) / (2.0 * k * t)
	# return tf.reduce_sum(A)


def build_SE(dimension, act_func, gamma, k, e, ILL, KG):
	tf.reset_default_graph()
	input_layer = get_se_input_layer(e, dimension)
	M = get_sparse_tensor(e, KG)
	hidden_layer = add_diag_layer(input_layer, dimension, M, act_func, dropout=0.0)
	output_layer = add_diag_layer(hidden_layer, dimension, M, None, dropout=0.0)
	loss = get_loss(output_layer, ILL, gamma, k)
	return output_layer, loss


def build_AE(attr, dimension, act_func, gamma, k, e, ILL, KG):
	tf.reset_default_graph()
	input_layer = get_ae_input_layer(attr)
	M = get_sparse_tensor(e, KG)
	hidden_layer = add_full_layer(input_layer, attr.shape[1], dimension, M, act_func, dropout=0.0)
	output_layer = add_diag_layer(hidden_layer, dimension, M, None, dropout=0.0)
	loss = get_loss(output_layer, ILL, gamma, k)
	return output_layer, loss


def training(output_layer, loss, learning_rate, epochs, ILL, e, k):
	# train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)  # optimizer can be changed
	print('initializing...')
	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)
	print('running...')
	J = []
	t = len(ILL)
	ILL = np.array(ILL)
	L = np.ones((t, k)) * (ILL[:, 0].reshape((t, 1)))
	neg_left = L.reshape((t * k,))
	L = np.ones((t, k)) * (ILL[:, 1].reshape((t, 1)))
	neg2_right = L.reshape((t * k,))
	for i in range(epochs):
		if i % 10 == 0:
			neg2_left = np.random.choice(e, t * k)
			neg_right = np.random.choice(e, t * k)
		# print(i)
		# print(sess.run(loss, feed_dict={"neg_left:0": neg_left,
		# 								"neg_right:0": neg_right,
		# 								"neg2_left:0": neg2_left,
		# 								"neg2_right:0": neg2_right}))
		sess.run(train_step, feed_dict={"neg_left:0": neg_left,
										"neg_right:0": neg_right,
										"neg2_left:0": neg2_left,
										"neg2_right:0": neg2_right})
		if (i + 1) % 20 == 0:
			th = sess.run(loss, feed_dict={"neg_left:0": neg_left,
										   "neg_right:0": neg_right,
										   "neg2_left:0": neg2_left,
										   "neg2_right:0": neg2_right})
			J.append(th)
			print('%d/%d' % (i + 1, epochs), 'epochs...')
	outvec = sess.run(output_layer)
	sess.close()
	return outvec, J


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

def get_ent2id(fns):
    ent2id = {}
    for fn in fns:
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                th = line[:-1].split('\t')
                ent2id[th[1]] = int(th[0])
    return ent2id


# The most frequent attributes are selected to save space
def loadattr(fns, e, ent2id):
    cnt = {}
    for fn in fns:
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                th = line[:-1].split('\t')
                if th[0] not in ent2id:
                    continue
                for i in range(1, len(th)):
                    if th[i] not in cnt:
                        cnt[th[i]] = 1
                    else:
                        cnt[th[i]] += 1
    fre = [(k, cnt[k]) for k in sorted(cnt, key=cnt.get, reverse=True)]
    attr2id = {}

    at = 1000

    if len(cnt) < at:
        at = len(cnt)

    for i in range(at):
        attr2id[fre[i][0]] = i
    attr = np.zeros((e, at), dtype=np.float32)
    for fn in fns:
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                th = line[:-1].split('\t')
                if th[0] in ent2id:
                    for i in range(1, len(th)):
                        if th[i] in attr2id:
                            attr[ent2id[th[0]]][attr2id[th[i]]] = 1.0
    return attr

def getsim_matrix_cosine(se_vec, ne_vec, test_pair):
    Lvec = tf.placeholder(tf.float32, [None, se_vec.shape[1]])
    Rvec = tf.placeholder(tf.float32, [None, se_vec.shape[1]])
    he = tf.nn.l2_normalize(Lvec, dim=-1)
    norm_e_em = tf.nn.l2_normalize(Rvec, dim=-1)
    aep = tf.matmul(he, tf.transpose(norm_e_em))

    Lvec_ne = tf.placeholder(tf.float32, [None, ne_vec.shape[1]])
    Rvec_ne = tf.placeholder(tf.float32, [None, ne_vec.shape[1]])
    he_n = tf.nn.l2_normalize(Lvec_ne, dim=-1)
    norm_e_em_n = tf.nn.l2_normalize(Rvec_ne, dim=-1)
    aep_n = tf.matmul(he_n, tf.transpose(norm_e_em_n))

    sess = tf.Session()
    Lv = np.array([se_vec[e1] for e1, e2 in test_pair])
    Lid_record = np.array([e1 for e1, e2 in test_pair])
    Rv = np.array([se_vec[e2] for e1, e2 in test_pair])
    Rid_record = np.array([e2 for e1, e2 in test_pair])

    Lv_ne = np.array([ne_vec[e1] for e1, e2 in test_pair])
    Rv_ne = np.array([ne_vec[e2] for e1, e2 in test_pair])
    aep = sess.run(aep, feed_dict = {Lvec: Lv, Rvec: Rv})
    aep_n = sess.run(aep_n, feed_dict = {Lvec_ne: Lv_ne, Rvec_ne: Rv_ne})
    aep = 1-aep
    aep_n = 1-aep_n

    return aep, aep_n


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