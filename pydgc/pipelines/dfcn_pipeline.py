# -*- coding: utf-8 -*-
from sklearn.decomposition import PCA
import scipy.sparse as sp
import numpy as np
import torch
from ..models import DFCN
from . import BasePipeline
from argparse import Namespace
from ..utils import perturb_data
from torch_geometric.utils import contains_self_loops, remove_self_loops, add_remaining_self_loops


def normalize(mx):
    row_sum = np.array(mx.sum(1))
    r_inv = np.power(row_sum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


class DFCNPipeline(BasePipeline):
    def __init__(self, args: Namespace):
        super(DFCNPipeline, self).__init__(args)

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
        edges = self.data.edge_index.numpy().T
        n = self.cfg.dataset.num_nodes
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(n, n), dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = adj + sp.eye(adj.shape[0])
        adj = normalize(adj)
        adj = sparse_mx_to_torch_sparse_tensor(adj)

        self.data.adj = adj

    def build_model(self):
        model = DFCN(self.logger, self.cfg)
        self.logger.model_info(model)
        return model
