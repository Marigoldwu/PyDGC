# -*- coding: utf-8 -*-
import scipy.sparse as sp
import numpy as np
import torch

from ..models import RGAE
from . import BasePipeline
from argparse import Namespace
from ..utils import perturb_data
from torch_geometric.utils import contains_self_loops, remove_self_loops, to_dense_adj, \
    add_remaining_self_loops


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)


class RGAEPipeline(BasePipeline):
    def __init__(self, args: Namespace):
        super().__init__(args.cfg_file_path, args.dataset_name)

    def augment_data(self):
        """Data augmentation"""
        self.data = perturb_data(self.data, self.cfg.dataset.augmentation)
        if hasattr(self.cfg.dataset.augmentation, 'add_self_loops'):
            if self.cfg.dataset.augmentation.add_self_loops:
                edge_index, _ = add_remaining_self_loops(self.data.edge_index, num_nodes=self.data.num_nodes)
                self.data.edge_index = edge_index
        adj = sp.csr_matrix(to_dense_adj(self.data.edge_index)[0])
        adj_norm = preprocess_graph(adj)
        features = sparse_to_tuple(sp.coo_matrix(self.data.x))
        pos_weight_orig = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        # norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
        adj_label = adj + sp.eye(adj.shape[0])
        adj_label = sparse_to_tuple(adj_label)
        adj_norm = torch.sparse_coo_tensor(torch.LongTensor(adj_norm[0].T), torch.FloatTensor(adj_norm[1]),
                                           torch.Size(adj_norm[2]))
        adj_label = torch.sparse_coo_tensor(torch.LongTensor(adj_label[0].T), torch.FloatTensor(adj_label[1]),
                                            torch.Size(adj_label[2]))
        features = torch.sparse_coo_tensor(torch.LongTensor(features[0].T), torch.FloatTensor(features[1]),
                                           torch.Size(features[2]))
        weight_mask_orig = adj_label.to_dense().view(-1) == 1
        weight_tensor_orig = torch.ones(weight_mask_orig.size(0))
        weight_tensor_orig[weight_mask_orig] = pos_weight_orig
        self.data.x = features
        self.data.adj = adj.tocoo()
        self.data.adj_label = adj_label
        self.data.adj_norm = adj_norm
        self.data.weight_tensor = weight_tensor_orig

    def build_model(self):
        model = RGAE(self.logger, self.cfg)
        self.logger.model_info(model)
        return model
