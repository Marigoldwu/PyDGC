# -*- coding: utf-8 -*-
from ..models import DGCLUSTER
from . import BasePipeline
from argparse import Namespace
from ..utils import perturb_data
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
import torch


class DGCLUSTERPipeline(BasePipeline):
    def __init__(self, args: Namespace):
        super().__init__(args.cfg_file_path, args.dataset_name)

    def augment_data(self):
        """Data augmentation"""
        self.data = perturb_data(self.data, self.cfg.dataset.augmentation)

        # transform = T.NormalizeFeatures()
        num_nodes = self.data.x.shape[0]
        num_edges = (self.data.edge_index.shape[1])
        labels = self.data.y.flatten()
        oh_labels = F.one_hot(labels, num_classes=max(labels) + 1)
        sparse_adj = sp.csr_matrix((np.ones(num_edges), self.data.edge_index.cpu().numpy()),
                                   shape=(num_nodes, num_nodes))
        degree = torch.tensor(sparse_adj.sum(axis=1)).squeeze().float().to(self.device)
        num_edges = int((self.data.edge_index.shape[1]) / 2)

        self.data.oh_labels = oh_labels
        self.data.sparse_adj = sparse_adj
        self.data.degree = degree
        self.cfg.dataset.num_edges = num_edges

        self.data.to(self.device)

    def build_model(self):
        model = DGCLUSTER(self.logger, self.cfg)
        self.logger.model_info(model)
        return model
