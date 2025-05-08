# -*- coding: utf-8 -*-
import torch
import numpy as np
from ..models import HSAN
from . import BasePipeline
from argparse import Namespace
from ..utils import perturb_data
from sklearn.decomposition import PCA
from torch_geometric.utils import to_dense_adj, add_remaining_self_loops


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
        norm_adj = np.matmul(np.matmul(sqrt_d_inv, adj_tmp), sqrt_d_inv)

    # non-symmetry normalize: D^{-1} A
    else:
        norm_adj = np.matmul(d_inv, adj_tmp)
    return norm_adj


def laplacian_filtering(A, X, t):
    A_tmp = A - torch.diag_embed(torch.diag(A))
    A_norm = normalize_adj(A_tmp, self_loop=True, symmetry=True).float()
    I = torch.eye(A.shape[0])
    L = I - A_norm
    for i in range(t):
        X = (I - L) @ X
    return X.float()


class HSANPipeline(BasePipeline):
    def __init__(self, args: Namespace):
        super().__init__(args.cfg_file_path, args.dataset_name)

    def augment_data(self):
        """Data augmentation"""
        self.data = perturb_data(self.data, self.cfg.dataset.augmentation)
        if hasattr(self.cfg.dataset.augmentation, 'add_self_loops'):
            if self.cfg.dataset.augmentation.add_self_loops:
                edge_index, _ = add_remaining_self_loops(self.data.edge_index, num_nodes=self.data.num_nodes)
                self.data.edge_index = edge_index
        self.data.adj = to_dense_adj(self.data.edge_index)[0]
        if self.cfg.model.dims.input_dim != -1:
            pca = PCA(n_components=self.cfg.model.dims.input_dim)
            self.data.x = torch.from_numpy(pca.fit_transform(self.data.x))

        self.data.x = laplacian_filtering(self.data.adj, self.data.x, self.cfg.train.t)

    def build_model(self):
        model = HSAN(self.logger, self.cfg)
        self.logger.model_info(model)
        return model
