# -*- coding: utf-8 -*-
import torch
import numpy as np
from torch import Tensor

from torch_geometric.data import Data
from torch_geometric.utils import dropout_edge, mask_feature, add_random_edge
from yacs.config import CfgNode as CN
from sklearn.preprocessing import normalize


def get_M(adj, t: int = 2):
    """
    calculate the matrix M by the equation:
        M=(B^1 + B^2 + ... + B^t) / t

    :param adj: the adjacency matrix
    :param t: default value is 2
    :return: M
    """
    if adj.device != torch.device("cpu"):
        adj = adj.cpu()
    if isinstance(adj, torch.Tensor):
        adj = adj.cpu().numpy()
    tran_prob = normalize(adj, norm="l1", axis=0)
    M_numpy = sum([np.linalg.matrix_power(tran_prob, i) for i in range(1, t + 1)]) / t
    return torch.Tensor(M_numpy)


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def diffusion_adj(adj, mode="ppr", transport_rate=0.2):
    """
    graph diffusion
    :param adj: input adj matrix
    :param mode: the mode of graph diffusion
    :param transport_rate: the transport rate
    - personalized page rank
    -
    :return: the graph diffusion
    """
    # add the self_loop
    adj_tmp = adj + np.eye(adj.shape[0])

    # calculate degree matrix and it's inverse matrix
    d = np.diag(adj_tmp.sum(0))
    d_inv = np.linalg.inv(d)
    sqrt_d_inv = np.sqrt(d_inv)

    # calculate norm adj
    norm_adj = np.matmul(np.matmul(sqrt_d_inv, adj_tmp), sqrt_d_inv)

    # calculate graph diffusion
    if mode == "ppr":
        diff_adj = transport_rate * np.linalg.inv((np.eye(d.shape[0]) - (1 - transport_rate) * norm_adj))
    else:
        diff_adj = None
    return diff_adj


def add_gaussian_noise(x: Tensor, mean=0, std_dev=0.1):
    """
    Add gaussian noise to x
    """
    noise = torch.normal(mean, std_dev, size=x.size())
    x = x + noise
    return x


def perturb_data(data: Data, cfg: CN):
    if hasattr(cfg, 'drop_edge'):
        drop_edge_rate = float(cfg.drop_edge) if float(cfg.drop_edge) < 0.99 else 0.99
        if cfg.drop_edge > 0:
            edge_index, edge_mask = dropout_edge(data.edge_index, drop_edge_rate)
            data.edge_index = edge_index
            data.edge_mask = edge_mask
    if hasattr(cfg, 'drop_feature'):
        drop_feature_rate = float(cfg.drop_feature) if float(cfg.drop_feature) < 0.99 else 0.99
        if cfg.drop_feature > 0:
            x, feature_mask = mask_feature(data.x, drop_feature_rate, mode='all')
            data.x = x
            data.feature_mask = feature_mask
    if hasattr(cfg, 'add_edge'):
        add_edge_rate = float(cfg.add_edge) if float(cfg.add_edge) < 0.99 else 0.99
        if cfg.add_edge > 0:
            edge_index, added_edges = add_random_edge(data.edge_index, add_edge_rate)
            data.edge_index = edge_index
            data.added_edges = added_edges
    if hasattr(cfg, 'add_noise'):
        noise_std = float(cfg.add_noise)
        if cfg.add_noise > 0:
            x = add_gaussian_noise(data.x, std_dev=noise_std)
            data.x = x
    return data


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def normalize_adj_torch(adj, symmetry=True):
    """
    Normalize the adjacency matrix.

    :param adj: Input adjacency matrix
    :param symmetry: Symmetry normalize or not
    :return norm_adj: The normalized adjacency matrix
    """
    # Calculate degree matrix and its inverse matrix
    d_inv = torch.diag(1 / torch.sum(adj, dim=1))

    # Symmetry normalize: D^(-0.5) A D^(-0.5)
    if symmetry:
        sqrt_d_inv = torch.sqrt(d_inv)
        norm_adj = torch.matmul(torch.matmul(sqrt_d_inv, adj), sqrt_d_inv)

    # Non-symmetry normalize: D^(-1) A
    else:
        norm_adj = torch.matmul(d_inv, adj)

    return norm_adj
