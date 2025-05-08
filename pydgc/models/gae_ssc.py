# -*- coding: utf-8 -*-
import os
import torch

from . import GAE
from torch import Tensor
from ..metrics import DGCMetric
from ..modules import SSCLayer
from .dgc_model import DGCModel
from typing import Tuple, List, Any
from torch_geometric.data import Data
from yacs.config import CfgNode as CN
from ..utils import validate_and_create_path


class GAESSC(DGCModel):

    def __init__(self, logger, cfg):
        super(GAESSC, self).__init__(logger, cfg)
        self.device = torch.device(cfg.device)
        self.gae = GAE(logger, cfg).to(self.device)
        self.ssc = SSCLayer(in_channels=self.cfg.model.dims[-1], out_channels=self.cfg.dataset.n_clusters, method='kl_div').to(self.device)
        self.loss_curve = []
        self.pretrain_loss_curve = []
        self.results = {}
        self.reset_parameters()

    def reset_parameters(self):
        self.gae.reset_parameters()
        self.ssc.reset_parameters()

    def forward(self, data) -> Any:
        hat_adj, embedding = self.gae(data)
        q = self.ssc(embedding)
        return hat_adj, q

    def loss(self, edge_index, hat_adj: Tensor, q: Tensor) -> Tensor:
        reconstruct_loss = self.gae.loss(edge_index, hat_adj)
        ssc_loss = self.ssc.loss(q, method='kl_div')
        loss_total = reconstruct_loss + float(self.cfg.train.alpha) * ssc_loss
        return loss_total

    def pretrain(self, data: Data, cfg: CN = None, flag: str = "PRETRAIN GAE"):
        flag += f'-{self.gae.cfg.model.gnn_type.upper()}'
        if cfg is None:
            cfg = self.cfg.train.pretrain
        self.pretrain_loss_curve = self.gae.train_model(data, cfg, flag)
        validate_and_create_path(cfg.dir)
        pretrain_file_name = os.path.join(cfg.dir, f'gae_{self.cfg.model.gnn_type}.pth')
        torch.save(self.gae.state_dict(), pretrain_file_name)

    def train_model(self, data: Data, cfg: CN = None, flag: str = "TRAIN GAE-SSC") -> List:
        flag += f'-{self.cfg.model.gnn_type.upper()}'
        if cfg is None:
            cfg = self.cfg.train
        # load pretrained gae model
        pretrain_file_name = os.path.join(cfg.pretrain.dir, f'gae_{self.cfg.model.gnn_type}.pth')
        if not os.path.exists(pretrain_file_name):
            self.pretrain(data, cfg.pretrain, flag='PRETRAIN GAE')
        self.gae.load_state_dict(torch.load(pretrain_file_name, map_location=self.device))

        optimizer = torch.optim.Adam(self.parameters(), lr=float(cfg.lr))

        # initialize ssc layer
        _, _, cluster_centers = self.gae.clustering(data)
        self.ssc.cluster_centers.data = cluster_centers.to(self.device)
        self.gae.evaluate(data)

        self.logger.flag(flag)
        # train
        for epoch in range(1, cfg.max_epoch+1):
            self.train()
            optimizer.zero_grad()
            hat_adj, q = self.forward(data)
            loss = self.loss(data.edge_index, hat_adj, q)
            loss.backward()
            optimizer.step()
            self.loss_curve.append(loss.item())
            self.logger.loss(epoch, loss)
            if self.cfg.evaluate.each:
                self.evaluate(data)
        return self.loss_curve

    def get_embedding(self, data) -> Tensor:
        with torch.no_grad():
            self.eval()
            return self.gae.get_embedding(data)

    def clustering(self, data) -> Tuple[Tensor, Tensor, Tensor]:
        embedding = self.get_embedding(data)
        labels_ = torch.from_numpy(self.ssc.get_q(embedding).detach().cpu().numpy().argmax(axis=1))
        clustering_centers = self.ssc.cluster_centers.data
        return embedding, labels_, clustering_centers

    def evaluate(self, data: Data):
        embedding, labels, clustering_centers = self.clustering(data)
        ground_truth = data.y.numpy()
        metric = DGCMetric(ground_truth, labels.numpy(), embedding, data.edge_index)
        metric.evaluate_one_epoch(self.logger, acc=True, nmi=True, ari=True, f1=True, hom=True, com=True, pur=True)
