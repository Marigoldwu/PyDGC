# -*- coding: utf-8 -*-
from sklearn.decomposition import PCA
import numpy as np
import torch
from ..models import DCRN
from . import BasePipeline
from argparse import Namespace
from ..utils import perturb_data
from torch_geometric.utils import contains_self_loops, remove_self_loops, to_dense_adj, \
    add_remaining_self_loops


def normalize_adj(adj, self_loop=True, symmetry=False):
    """
    normalize the adj matrix
    :param adj: input adj matrix
    :param self_loop: if add the self loop or not
    :param symmetry: symmetry normalize or not
    :return: the normalized adj matrix
    """
    # add the self_loop
    if self_loop:
        adj_tmp = adj + np.eye(adj.shape[0])
    else:
        adj_tmp = adj

    # calculate degree matrix and it's inverse matrix
    d = np.diag(adj_tmp.sum(0))
    d_inv = np.linalg.inv(d)

    # symmetry normalize: D^{-0.5} A D^{-0.5}
    if symmetry:
        sqrt_d_inv = np.sqrt(d_inv)
        norm_adj = np.matmul(np.matmul(sqrt_d_inv, adj_tmp), adj_tmp)

    # non-symmetry normalize: D^{-1} A
    else:
        norm_adj = np.matmul(d_inv, adj_tmp)

    return norm_adj


def diffusion_adj(adj, transport_rate=0.2):
    """
    graph diffusion
    :param adj: input adj matrix
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
    diff_adj = transport_rate * np.linalg.inv((np.eye(d.shape[0]) - (1 - transport_rate) * norm_adj))

    return diff_adj


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


class DCRNPipeline(BasePipeline):
    def __init__(self, args: Namespace):
        super(DCRNPipeline, self).__init__(args)

    def augment_data(self):
        """Data augmentation"""
        self.data = perturb_data(self.data, self.cfg.dataset.augmentation)
        pca = PCA(n_components=self.cfg.dataset.augmentation.pca_dim)
        self.data.x = torch.from_numpy(pca.fit_transform(self.data.x)).float()

        if hasattr(self.cfg.dataset.augmentation, 'add_self_loops'):
            if self.cfg.dataset.augmentation.add_self_loops:
                edge_index, _ = add_remaining_self_loops(self.data.edge_index, num_nodes=self.data.num_nodes)
                self.data.edge_index = edge_index
        if contains_self_loops(self.data.edge_index):
            self.data.edge_index = remove_self_loops(self.data.edge_index)[0]

        A = to_dense_adj(self.data.edge_index)[0].numpy()
        self.data.A_norm = normalize_adj(A, self_loop=True, symmetry=False)
        self.data.Ad = diffusion_adj(A, transport_rate=self.cfg.dataset.alpha_value)
        self.data.adj = A

    def build_model(self):
        model = DCRN(self.logger, self.cfg)
        self.logger.model_info(model)
        return model
