import numpy as np
from fml.functional import sinkhorn
import torch

def optimize(aep_fuse, score_strategy, args):
    if score_strategy == "csls":
        if args.mode == "mul":
            aep_fuse = csls_sim(aep_fuse, args.multik)
        else:
            aep_fuse = csls_sim(aep_fuse, 1)
    elif score_strategy == "rinf":
        # the lower the better, dis!!!
        if args.mode == "mul":
            aep_fuse = 1 - recip_mul(aep_fuse, args.multik)
        else:
            aep_fuse = 1 - recip(aep_fuse)
    elif score_strategy == "sinkhorn":
        if args.mode == "mul":
            dev = "cpu"
            aep_fuse = matrix_sinkhorn(torch.tensor(1 - aep_fuse, device=dev), args.sinkhornIte).cpu().detach().numpy()
            print(aep_fuse)
        else:
            # input to sinkhorn is a distance matrix
            aep_fuse = matrix_sinkhorn(torch.tensor(1 - aep_fuse, device="cpu"), 100).cpu().detach().numpy()  # cuda
            # output of sinkhorn is a similarity matrix
            print(aep_fuse)
    else:
        aep_fuse = aep_fuse
    return aep_fuse

def calculate_nearest_k(sim_mat, k):
    sorted_mat = -np.partition(-sim_mat, k + 1, axis=1)  # -np.sort(-sim_mat1)
    nearest_k = sorted_mat[:, 0:k]
    return np.mean(nearest_k, axis=1, keepdims=True)

def csls_sim(sim_mat, k):
    nearest_values1 = calculate_nearest_k(sim_mat, k)
    nearest_values2 = calculate_nearest_k(sim_mat.T, k)
    csls_sim_mat = 2 * sim_mat - nearest_values1 - nearest_values2.T
    del sim_mat
    return csls_sim_mat

# reciprocal inference
from scipy.stats import rankdata
def recip(string_mat):
    max_value = np.max(string_mat, axis=0)
    max_value[max_value == 0.0] = 1.0
    a = string_mat - max_value + 1
    a = -a
    a_rank = rankdata(a, axis=1)
    del a
    max_value = np.max(string_mat, axis=1)
    max_value[max_value == 0.0] = 1.0
    b = (string_mat.T - max_value) + 1
    del string_mat
    del max_value
    b = -b
    b_rank = rankdata(b, axis=1)
    del b
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
    from scipy.stats import rankdata
    a_rank = rankdata(-a, axis=1)
    del a
    b_rank = rankdata(-b, axis=1)
    del b
    recip_sim = (a_rank + b_rank.T) / 2.0
    del a_rank
    del b_rank
    return recip_sim

def view3(x):
    if x.dim() == 3:
        return x
    return x.view(1, x.size(0), -1)

def view2(x):
    if x.dim() == 2:
        return x
    return x.view(-1, x.size(-1))

def matrix_sinkhorn(pred_or_m, iter):
    device = pred_or_m.device
    M = view3(pred_or_m).to(torch.float32)
    m, n = tuple(pred_or_m.size())
    a = torch.ones([1, m], requires_grad=False, device=device)
    b = torch.ones([1, n], requires_grad=False, device=device)
    P = sinkhorn(a, b, M, 1e-3, max_iters=iter, stop_thresh=1e-3) #max_iters=300
    return view2(P)