# -*- coding: utf-8 -*-
import torch
from argparse import Namespace

from torch_geometric.utils import add_self_loops

from . import BasePipeline
from ..utils import perturb_data
from ..models import NS4GC


class NS4GCPipeline(BasePipeline):
    """NS4GC pipeline.

    Args:
        args (Namespace): Arguments.
    """
    def __init__(self, args: Namespace):
        super(NS4GCPipeline, self).__init__(args)

    def augment_data(self):
        self.data = perturb_data(self.data, self.cfg.dataset.augmentation)
        x, edge_index = self.data.x, self.data.edge_index
        if self.dataset_name == "DBLP":
            edge_index = add_self_loops(edge_index)[0]
        N, E = self.data.num_nodes, (edge_index.shape[1])
        A = torch.sparse_coo_tensor(edge_index, torch.ones(E), size=(N, N))
        src, dst = edge_index[0], edge_index[1]
        mask = torch.full(A.size(), True)
        mask[src, dst] = False
        mask.fill_diagonal_(False)
        self.data.edge_index = edge_index
        self.data.A = A
        self.data.mask = mask

    def build_model(self):
        model = NS4GC(self.logger, self.cfg)
        self.logger.model_info(model)
        return model
