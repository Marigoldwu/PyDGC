# -*- coding: utf-8 -*-
import os
import torch
import torch.nn.functional as F

from . import AE
from ..metrics import DGCMetric
from torch import Tensor
from .dgc_model import DGCModel
from pydgc.modules import SSCLayer
from typing import Tuple, List, Any
from yacs.config import CfgNode as CN
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from ..utils import Logger, target_distribution, validate_and_create_path


class SDCN(DGCModel):
    def __init__(self, logger: Logger, cfg: CN):
        super(SDCN, self).__init__(logger, cfg)
        self.ae = AE(logger, cfg)
        dims = self.cfg.model.dims
        dims.insert(0, self.cfg.dataset.num_features)
        self.gcn_1 = GCNConv(dims[0], dims[1], add_self_loops=cfg.dataset.augmentation.add_self_loops).to(self.device)
        self.gcn_2 = GCNConv(dims[1], dims[2], add_self_loops=cfg.dataset.augmentation.add_self_loops).to(self.device)
        self.gcn_3 = GCNConv(dims[2], dims[3], add_self_loops=cfg.dataset.augmentation.add_self_loops).to(self.device)
        self.gcn_4 = GCNConv(dims[3], dims[-1], add_self_loops=cfg.dataset.augmentation.add_self_loops).to(self.device)
        self.gcn_5 = GCNConv(dims[-1], cfg.dataset.n_clusters, add_self_loops=cfg.dataset.augmentation.add_self_loops).to(self.device)

        self.ssc = SSCLayer(in_channels=dims[-1], out_channels=self.cfg.dataset.n_clusters, method='kl_div').to(self.device)

        self.loss_curve = []
        self.pretrain_loss_curve = []

        self.reset_parameters()

    def reset_parameters(self):
        self.ae.reset_parameters()
        self.gcn_1.reset_parameters()
        self.gcn_2.reset_parameters()
        self.gcn_3.reset_parameters()
        self.gcn_4.reset_parameters()
        self.gcn_5.reset_parameters()
        self.ssc.reset_parameters()

    def forward(self, data: Data, sigma: float = 0.5) -> Any:
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        encodes, decodes = self.ae.forward(x)
        hat_x = decodes[-1]

        h = F.relu(self.gcn_1(x, edge_index))
        h = F.relu(self.gcn_2((1 - sigma) * h + sigma * encodes[0], edge_index))
        h = F.relu(self.gcn_3((1 - sigma) * h + sigma * encodes[1], edge_index))
        h = F.relu(self.gcn_4((1 - sigma) * h + sigma * encodes[2], edge_index))
        embedding = self.gcn_5((1 - sigma) * h + sigma * encodes[-1], edge_index)
        predict = F.softmax(embedding, dim=1)

        q = self.ssc(encodes[-1])
        return predict, embedding, hat_x, q

    def loss(self, x, hat_x, q, pred) -> Tensor:
        reconstruct_loss = self.ae.loss(x, hat_x)
        ssc_loss = self.ssc.loss(q, method='kl_div')
        p = target_distribution(q.detach())
        ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
        alpha = float(self.cfg.train.alpha)
        beta = float(self.cfg.train.beta)
        loss_total = reconstruct_loss + alpha * ssc_loss + beta * ce_loss
        return loss_total

    def pretrain(self, data: Data, cfg: CN = None, flag: str = "PRETRAIN AE"):
        if cfg is None:
            cfg = self.cfg.train.pretrain
        self.pretrain_loss_curve = self.ae.train_model(data, cfg, flag)
        validate_and_create_path(cfg.dir)
        pretrain_file_name = os.path.join(cfg.dir, f'ae.pth')
        torch.save(self.ae.state_dict(), pretrain_file_name)

    def train_model(self, data: Data, cfg: CN = None, flag: str = "TRAIN SDCN") -> List:
        if cfg is None:
            cfg = self.cfg.train
        # load pretrained ae model
        pretrain_file_name = os.path.join(cfg.pretrain.dir, f'ae.pth')

        if not os.path.exists(pretrain_file_name):
            self.pretrain(data, cfg.pretrain, flag='PRETRAIN AE')
        self.ae.load_state_dict(torch.load(pretrain_file_name, map_location='cpu'))

        optimizer = torch.optim.Adam(self.parameters(), lr=float(cfg.lr))

        # initialize ssc layer
        _, _, cluster_centers = self.ae.clustering(data)
        self.ssc.cluster_centers.data = cluster_centers.to(self.device)
        self.ae.evaluate(data)

        self.logger.flag(flag)
        # train
        for epoch in range(1, cfg.max_epoch + 1):
            self.train()
            optimizer.zero_grad()
            predict, embedding, hat_x, q = self.forward(data, cfg.sigma)
            loss = self.loss(data.x, hat_x, q, predict)
            loss.backward()
            optimizer.step()
            self.loss_curve.append(loss.item())
            self.logger.loss(epoch, loss)
            if self.cfg.evaluate.each:
                self.evaluate(data)
        return self.loss_curve

    def get_embedding(self, data) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            self.eval()
            predict, embedding, _, _ = self.forward(data, self.cfg.train.sigma)
            return predict, embedding

    def clustering(self, data) -> Tuple[Tensor, Tensor, Tensor]:
        predict, embedding = self.get_embedding(data)
        labels_ = torch.from_numpy(predict.detach().cpu().numpy().argmax(axis=1))
        clustering_centers = self.ssc.cluster_centers.data
        return embedding, labels_, clustering_centers

    def evaluate(self, data: Data):
        embedding, labels, clustering_centers = self.clustering(data)
        ground_truth = data.y.numpy()
        metric = DGCMetric(ground_truth, labels.numpy(), embedding, data.edge_index)
        metric.evaluate_one_epoch(self.logger, acc=True, nmi=True, ari=True, f1=True, hom=True, com=True, pur=True, sc=True, gre=True)
