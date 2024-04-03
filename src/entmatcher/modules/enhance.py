import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import faiss

import scipy.sparse as sp
from scipy.sparse import lil_matrix
import time
tf.compat.v1.enable_eager_execution()

def gen(vec, dataset, algo):
    if algo == 'tfp':
        node_size, rel_size, ent_tuple, triples_idx, ent_ent, ent_ent_val, rel_ent, ent_rel = load_graph(dataset)
        Triple_FP = TripleFeaturePropagation(tf.cast(vec, "float32"), train_pair=dataset.train)
        features = Triple_FP.propagation(node_size, rel_size, ent_tuple, triples_idx, ent_ent, ent_ent_val, rel_ent, ent_rel)
        return features

#### TFP
def normalize_adj(adj):
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -1).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj)

def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo().astype(np.float32)
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)

def batch_sparse_matmul(sparse_tensor, dense_tensor, batch_size=128, save_mem=False):
    results = []
    for i in range(dense_tensor.shape[-1] // batch_size + 1):
        temp_result = tf.sparse.sparse_dense_matmul(sparse_tensor, dense_tensor[:, i * batch_size:(i + 1) * batch_size])
        if save_mem:
            temp_result = temp_result.numpy()
        results.append(temp_result)
    if save_mem:
        return np.concatenate(results, -1)
    else:
        return K.concatenate(results, -1)
    
def random_projection(x, out_dim):
    random_vec = K.l2_normalize(tf.random.normal((x.shape[-1], out_dim), mean=0, stddev=(1 / out_dim) ** 0.5), axis=-1)
    return K.dot(x, random_vec)

def load_graph(d):
    triples = []
    for kg in [d.KG1, d.KG2]:
        for h, r, t in kg:
            triples.append([h, t, 2 * r])
            triples.append([t, h, 2 * r + 1])
    triples = np.unique(triples, axis=0)
    node_size, rel_size = np.max(triples) + 1, np.max(triples[:, 2]) + 1

    ent_tuple, triples_idx = [], []
    ent_ent_s, rel_ent_s, ent_rel_s = {}, set(), set()
    last, index = (-1, -1), -1

    adj_matrix = lil_matrix((node_size, node_size))
    for h, r, t in triples:
        adj_matrix[h, t] = 1
        adj_matrix[t, h] = 1

    for i in range(node_size):
        ent_ent_s[(i, i)] = 0

    for h, t, r in triples:
        ent_ent_s[(h, h)] += 1
        ent_ent_s[(t, t)] += 1

        if (h, t) != last:
            last = (h, t)
            index += 1
            ent_tuple.append([h, t])
            ent_ent_s[(h, t)] = 0

        triples_idx.append([index, r])
        ent_ent_s[(h, t)] += 1
        rel_ent_s.add((r, h))
        ent_rel_s.add((t, r))

    ent_tuple = np.array(ent_tuple)
    triples_idx = np.unique(np.array(triples_idx), axis=0)

    ent_ent = np.unique(np.array(list(ent_ent_s.keys())), axis=0)
    ent_ent_val = np.array([ent_ent_s[(x, y)] for x, y in ent_ent]).astype("float32")
    rel_ent = np.unique(np.array(list(rel_ent_s)), axis=0)
    ent_rel = np.unique(np.array(list(ent_rel_s)), axis=0)

    # graph_data = [node_size, rel_size, ent_tuple, triples_idx, ent_ent, ent_ent_val, rel_ent, ent_rel]
    graph_data = [node_size, rel_size, ent_tuple, triples_idx, ent_ent, ent_ent_val, rel_ent, ent_rel]
    return graph_data

class TripleFeaturePropagation:

    def __init__(self, initial_feature, train_pair=None):
        self.train_pair = train_pair
        self.initial_feature = initial_feature

    def propagation(self, node_size, rel_size, ent_tuple, triples_idx, ent_ent, ent_ent_val, rel_ent, ent_rel, rel_dim=512, mini_dim=16):

        ent_feature = self.initial_feature

        rel_feature = tf.zeros((rel_size, ent_feature.shape[-1]))

        ent_ent_graph = sp.coo_matrix((ent_ent_val, ent_ent.transpose()), shape=(node_size, node_size))
        ent_ent_graph = normalize_adj(ent_ent_graph)
        ent_ent_graph = convert_sparse_matrix_to_sparse_tensor(ent_ent_graph)

        rel_ent_graph = sp.coo_matrix((K.ones(rel_ent.shape[0]), rel_ent.transpose()), shape=(rel_size, node_size))
        rel_ent_graph = normalize_adj(rel_ent_graph)
        rel_ent_graph = convert_sparse_matrix_to_sparse_tensor(rel_ent_graph)

        ent_rel_graph = sp.coo_matrix((K.ones(ent_rel.shape[0]), ent_rel.transpose()), shape=(node_size, rel_size))
        ent_rel_graph = normalize_adj(ent_rel_graph)
        ent_rel_graph = convert_sparse_matrix_to_sparse_tensor(ent_rel_graph)

        ent_list, rel_list = [ent_feature], [rel_feature]
        start_time = time.time()
        for i in range(1):  # Dual-AMN iteration: 11:81.59, 12:81.62, 13:81.6, .
            new_rel_feature = batch_sparse_matmul(rel_ent_graph, ent_feature)
            new_rel_feature = tf.nn.l2_normalize(new_rel_feature, axis=-1)

            new_ent_feature = batch_sparse_matmul(ent_ent_graph, ent_feature)
            new_ent_feature = new_ent_feature.numpy()

            # ### Keeping stationary for aligned pairs ###
            if self.train_pair.any():
                ori_feature = self.initial_feature.numpy()
                new_ent_feature[self.train_pair[:, 0]] = ori_feature[self.train_pair[:, 0]]
                new_ent_feature[self.train_pair[:, 1]] = ori_feature[self.train_pair[:, 1]]

            new_ent_feature += batch_sparse_matmul(ent_rel_graph, rel_feature)
            new_ent_feature = tf.nn.l2_normalize(new_ent_feature, axis=-1)

            ent_feature = new_ent_feature
            rel_feature = new_rel_feature
            ent_list.append(ent_feature)
            rel_list.append(rel_feature)

        ent_feature = K.l2_normalize(K.concatenate(ent_list, 1), -1)
        rel_feature = K.l2_normalize(K.concatenate(rel_list, 1), -1)
        rel_feature = random_projection(rel_feature, rel_dim)


        batch_size = ent_feature.shape[-1] // mini_dim
        sparse_graph = tf.SparseTensor(indices=triples_idx, values=K.ones(triples_idx.shape[0]),
                                       dense_shape=(np.max(triples_idx) + 1, rel_size))
        adj_value = batch_sparse_matmul(sparse_graph, rel_feature)

        features_list = []
        for batch in range(rel_dim // batch_size + 1):
            temp_list = []
            for head in range(batch_size):
                if batch * batch_size + head >= rel_dim:
                    break
                sparse_graph = tf.SparseTensor(indices=ent_tuple, values=adj_value[:, batch * batch_size + head],
                                               dense_shape=(node_size, node_size))
                feature = batch_sparse_matmul(sparse_graph, random_projection(ent_feature, mini_dim))
                temp_list.append(feature)
            if len(temp_list):
                features_list.append(K.concatenate(temp_list, -1).numpy())
        features = np.concatenate(features_list, axis=-1)
        faiss.normalize_L2(features)

        #### Test the reconstructed entity feature ####
        features = np.concatenate([ent_feature, features], axis=-1)
        return features