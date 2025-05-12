# -*- coding: utf-8 -*-
import torch
from argparse import Namespace

from . import BasePipeline
from ..utils import perturb_data
from ..models import NS4GC


class NS4GCPipeline(BasePipeline):
    def __init__(self, args: Namespace):
        super(NS4GCPipeline, self).__init__(args)

    def augment_data(self):
        """Data augmentation"""
        self.data = perturb_data(self.data, self.cfg.dataset.augmentation)
        x, edge_index = self.data.x, self.data.edge_index
        N, E = self.data.num_nodes, self.data.num_edges
        A = torch.sparse_coo_tensor(edge_index, torch.ones(E), size=(N, N))
        src, dst = edge_index[0], edge_index[1]
        mask = torch.full(A.size(), True)
        mask[src, dst] = False
        mask.fill_diagonal_(False)
        self.data.A = A
        self.data.mask = mask

    def build_model(self):
        model = NS4GC(self.logger, self.cfg)
        self.logger.model_info(model)
        return model
