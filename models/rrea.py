import warnings
warnings.filterwarnings('ignore')
import os
import random
import keras
from tqdm import *
import numpy as np
from rrea_utils import *
from rrea_CSLS import *
import tensorflow as tf
import keras.backend as K
from keras.layers import *
from rrea_layer import NR_GraphAttention
from keras import activations, constraints, initializers, regularizers

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.compat.v1.reset_default_graph()
from tensorflow import optimizers

class TokenEmbedding(keras.layers.Embedding):
    """Embedding layer with weights returned."""
    def compute_output_shape(self, input_shape):
        return self.input_dim, self.output_dim
    def compute_mask(self, inputs, mask=None):
        return None
    def call(self, inputs):
        return self.embeddings

def get_embedding():
    inputs = [adj_matrix, r_index, r_val, rel_matrix, ent_matrix]
    inputs = [np.expand_dims(item, axis=0) for item in inputs]
    return get_emb.predict_on_batch(inputs)

def test(wrank=None):
    vec = get_embedding()
    return get_hits(vec, dev_pair, wrank=wrank)

def CSLS_test(thread_number=16, csls=10, accurate=True):
    vec = get_embedding()
    Lvec = np.array([vec[e1] for e1, e2 in dev_pair])
    Rvec = np.array([vec[e2] for e1, e2 in dev_pair])
    Lvec = Lvec / np.linalg.norm(Lvec, axis=-1, keepdims=True)
    Rvec = Rvec / np.linalg.norm(Rvec, axis=-1, keepdims=True)
    eval_alignment_by_sim_mat(Lvec, Rvec, [1, 5, 10], thread_number, csls=csls, accurate=accurate)
    np.save('../data/' + args.data_dir + '/vec-new.npy', vec)
    return None

def get_train_set(batch_size):
    negative_ratio = batch_size // len(train_pair) + 1
    train_set = np.reshape(np.repeat(np.expand_dims(train_pair, axis=0), axis=0, repeats=negative_ratio),newshape=(-1, 2))
    np.random.shuffle(train_set)
    train_set = train_set[:batch_size]
    train_set = np.concatenate([train_set, np.random.randint(0, node_size, train_set.shape)], axis=-1)
    return train_set

def get_trgat(node_size, rel_size, node_hidden, rel_hidden, triple_size, n_attn_heads=2, dropout_rate=0, gamma=3, lr=0.005, depth=2):
    adj_input = Input(shape=(None, 2))
    index_input = Input(shape=(None, 2), dtype='int64')
    val_input = Input(shape=(None,))
    rel_adj = Input(shape=(None, 2))
    ent_adj = Input(shape=(None, 2))
    ent_emb = TokenEmbedding(node_size, node_hidden, trainable=True)(
        val_input)  # do not care about the input, just obtain embeddings..
    rel_emb = TokenEmbedding(rel_size, node_hidden, trainable=True)(val_input)
    def avg(tensor, size):
        adj = K.cast(K.squeeze(tensor[0], axis=0), dtype="int64")
        adj = tf.SparseTensor(indices=adj, values=tf.ones_like(adj[:, 0], dtype='float32'),dense_shape=(node_size, size))
        adj = tf.sparse_softmax(adj)
        return tf.sparse_tensor_dense_matmul(adj, tensor[1])
    opt = [rel_emb, adj_input, index_input, val_input]
    ent_feature = Lambda(avg, arguments={'size': node_size})([ent_adj, ent_emb])
    rel_feature = Lambda(avg, arguments={'size': rel_size})([rel_adj, rel_emb])
    encoder = NR_GraphAttention(node_size, activation="relu",
                                rel_size=rel_size,
                                depth=depth,
                                attn_heads=n_attn_heads,
                                triple_size=triple_size,
                                attn_heads_reduction='average',
                                dropout_rate=dropout_rate)
    out_feature = Concatenate(-1)([encoder([ent_feature] + opt), encoder([rel_feature] + opt)])
    out_feature = Dropout(dropout_rate)(out_feature)
    alignment_input = Input(shape=(None, 4))
    find = Lambda(lambda x: K.gather(reference=x[0], indices=K.cast(K.squeeze(x[1], axis=0), 'int32')))([out_feature, alignment_input])

    def align_loss(tensor):
        def _cosine(x):
            dot1 = K.batch_dot(x[0], x[1], axes=1)
            dot2 = K.batch_dot(x[0], x[0], axes=1)
            dot3 = K.batch_dot(x[1], x[1], axes=1)
            max_ = K.maximum(K.sqrt(dot2 * dot3), K.epsilon())
            return dot1 / max_
        def l1(ll, rr):
            return K.sum(K.abs(ll - rr), axis=-1, keepdims=True)
        def l2(ll, rr):
            return K.sum(K.square(ll - rr), axis=-1, keepdims=True)
        l, r, fl, fr = [tensor[:, 0, :], tensor[:, 1, :], tensor[:, 2, :], tensor[:, 3, :]]
        loss = K.relu(gamma + l1(l, r) - l1(l, fr)) + K.relu(gamma + l1(l, r) - l1(fl, r))
        return tf.reduce_sum(loss, keep_dims=True) / (batch_size)

    loss = Lambda(align_loss)(find)
    inputs = [adj_input, index_input, val_input, rel_adj, ent_adj]
    train_model = keras.Model(inputs=inputs + [alignment_input], outputs=loss)
    #原来：train_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer=keras.optimizers.rmsprop(lr))
    train_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer=optimizers.RMSprop(lr))
    feature_model = keras.Model(inputs=inputs, outputs=out_feature)
    return train_model, feature_model

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="zh_en", required=False, help="input dataset file directory") #  en_fr_15k_V1 1hop dbp_wd_100
    args = parser.parse_args()
    print(args)

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    lang = args.data_dir
    train_pair, dev_pair, adj_matrix, r_index, r_val, adj_features, rel_features = load_data('../data/%s/' % lang)
    adj_matrix = np.stack(adj_matrix.nonzero(), axis=1)  # convert the sparse matrix to the connection matrix
    rel_matrix, rel_val = np.stack(rel_features.nonzero(),
                                   axis=1), rel_features.data  # convert the sparse matrix to the connection matrix with rel values..
    ent_matrix, ent_val = np.stack(adj_features.nonzero(), axis=1), adj_features.data

    node_size = adj_features.shape[0]
    rel_size = rel_features.shape[1]
    triple_size = len(adj_matrix)
    batch_size = node_size  # num of triples + self_loops!!!

    model, get_emb = get_trgat(dropout_rate=0.30, node_size=node_size, rel_size=rel_size, n_attn_heads=1, depth=2,
                               gamma=3, node_hidden=100, rel_hidden=100, triple_size=triple_size)
    model.summary()
    initial_weights = model.get_weights()

    rest_set_1 = [e1 for e1, e2 in dev_pair]
    rest_set_2 = [e2 for e1, e2 in dev_pair]
    np.random.shuffle(rest_set_1)
    np.random.shuffle(rest_set_2)
    # we aim to get the embeddings here
    epoch = 1200
    # for turn in range(5):
    #     print("iteration %d start." % turn)
    for i in trange(epoch):
        train_set = get_train_set(batch_size)
        inputs = [adj_matrix, r_index, r_val, rel_matrix, ent_matrix, train_set]
        inputs = [np.expand_dims(item, axis=0) for item in inputs]
        model.train_on_batch(inputs, np.zeros((1, 1)))
        if i % 300 == 299:
            CSLS_test()

