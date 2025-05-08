# -*- coding: utf-8 -*-
import numpy as np
import scipy.sparse as sp
from argparse import Namespace

import torch
from torch_geometric.utils import contains_self_loops, remove_self_loops, to_dense_adj, add_remaining_self_loops

from . import BasePipeline
from ..models import CCGC
from ..utils import perturb_data


def preprocess_graph(adj, layer, norm='sym', renorm=True):
    adj = sp.coo_matrix(adj)
    ident = sp.eye(adj.shape[0])
    if renorm:
        adj_ = adj + ident
    else:
        adj_ = adj

    row_sum = np.array(adj_.sum(1))

    if norm == 'sym':
        degree_mat_inv_sqrt = sp.diags(np.power(row_sum, -0.5).flatten())
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        laplacian = ident - adj_normalized
    elif norm == 'left':
        degree_mat_inv_sqrt = sp.diags(np.power(row_sum, -1.).flatten())
        adj_normalized = degree_mat_inv_sqrt.dot(adj_).tocoo()
        laplacian = ident - adj_normalized
    reg = [1] * layer

    adjs = []
    for i in range(len(reg)):
        adjs.append(ident - (reg[i] * laplacian))

    return adjs


class CCGCPipeline(BasePipeline):
    def __init__(self, args: Namespace):
        super().__init__(args.cfg_file_path, args.dataset_name)

    def augment_data(self):
        """Data augmentation"""
        self.data = perturb_data(self.data, self.cfg.dataset.augmentation)
        if contains_self_loops(self.data.edge_index):
            self.data.edge_index = remove_self_loops(self.data.edge_index)[0]
        if hasattr(self.cfg.dataset.augmentation, 'add_self_loops'):
            if self.cfg.dataset.augmentation.add_self_loops:
                edge_index, _ = add_remaining_self_loops(self.data.edge_index, num_nodes=self.data.num_nodes)
                self.data.edge_index = edge_index
        self.data.adj = to_dense_adj(self.data.edge_index)[0]
        # Laplacian Smoothing
        adj_norm_s = preprocess_graph(self.data.adj, self.cfg.train.t, norm='sym', renorm=True)
        smooth_fea = sp.csr_matrix(self.data.x).toarray()
        for a in adj_norm_s:
            smooth_fea = a.dot(smooth_fea)
        self.data.x = torch.FloatTensor(smooth_fea)

    def build_model(self):
        model = CCGC(self.logger, self.cfg)
        self.logger.model_info(model)
        return model
