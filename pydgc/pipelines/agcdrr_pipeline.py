# -*- coding: utf-8 -*-
import torch
from ..models import AGCDRR
from . import BasePipeline
from argparse import Namespace
from ..utils import perturb_data, normalize_adj_torch
from sklearn.decomposition import PCA
from torch_geometric.utils import to_dense_adj, remove_self_loops, add_remaining_self_loops


class AGCDRRPipeline(BasePipeline):
    def __init__(self, args: Namespace):
        super().__init__(args.cfg_file_path, args.dataset_name)

    def augment_data(self):
        """Data augmentation"""
        self.data = perturb_data(self.data, self.cfg.dataset.augmentation)

        pca = PCA(n_components=self.cfg.dataset.augmentation.pca_dim)
        self.data.x = torch.from_numpy(pca.fit_transform(self.data.x)).float()

        if hasattr(self.cfg.dataset.augmentation, 'add_self_loops'):
            if self.cfg.dataset.augmentation.add_self_loops:
                edge_index, _ = add_remaining_self_loops(self.data.edge_index, num_nodes=self.data.num_nodes)
                self.data.edge_index = edge_index
        self.data.adj = to_dense_adj(self.data.edge_index)[0]
        self.data.adj = normalize_adj_torch(self.data.adj, symmetry=False).to(self.device)
        self.data.edge_index = remove_self_loops(self.data.edge_index)[0]

    def build_model(self):
        model = AGCDRR(self.logger, self.cfg)
        self.logger.model_info(model)
        return model
