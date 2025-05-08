# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F

from . import DGCModel
from torch import Tensor
from ..utils import Logger
from typing import List, Tuple
from ..metrics import DGCMetric
from sklearn.cluster import KMeans
from ..clusterings import KMeansGPU
from ..datasets import LoadAttribute
from torch_geometric.data import Data
from yacs.config import CfgNode as CN
from torch.utils.data import DataLoader
from ..modules import MLPEncoder, MLPDecoder


class AE(DGCModel):

    def __init__(self, logger: Logger, cfg: CN):
        super(AE, self).__init__(logger, cfg)
        self.encoder = MLPEncoder(self.cfg.model.dims, self.cfg.model.act, self.cfg.model.act_last).to(self.device)
        self.decoder = MLPDecoder(self.cfg.model.dims, self.cfg.model.act, self.cfg.model.act_last).to(self.device)

        self.loss_curve = []

        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()

    def forward(self, x) -> Tuple[List[Tensor], List[Tensor]]:
        x = x.to(self.device)
        encodes = self.encoder(x)
        decodes = self.decoder(encodes[-1])
        return encodes, decodes

    def loss(self, x: Tensor, hat_x: Tensor) -> Tensor:
        x = x.to(self.device)
        loss = F.mse_loss(hat_x, x)
        return loss

    def train_model(self, data: Data, cfg: CN = None, flag: str = "TRAIN AE"):
        # when ae is trained in pre-training mode, cfg.pretrain must be input as parameter
        attribute = LoadAttribute(data.x)
        train_loader = DataLoader(attribute, batch_size=256, shuffle=True)
        if cfg is None:
            cfg = self.cfg.train
        self.logger.flag(flag)

        optimizer = torch.optim.Adam(self.parameters(), lr=float(cfg.lr))
        for epoch in range(1, cfg.max_epoch + 1):
            self.train()
            loss_sum = torch.tensor(0.0)
            for batch_idx, (x, _) in enumerate(train_loader):
                optimizer.zero_grad()
                _, decodes = self.forward(x)
                loss = self.loss(x, decodes[-1])
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
            self.loss_curve.append(loss_sum.item())
            self.logger.loss(epoch, loss_sum)
            if self.cfg.evaluate.each:
                self.evaluate(data)
        return self.loss_curve

    def get_embedding(self, x: Tensor) -> Tensor:
        x = x.to(self.device)
        with torch.no_grad():
            self.eval()
            encodes = self.encoder(x)
            return encodes[-1]

    def clustering(self, data: Data, method: str = 'kmeans_gpu') -> Tuple[Tensor, Tensor, Tensor]:
        embedding = self.get_embedding(data.x)
        if method == 'kmeans_gpu':
            labels_, clustering_centers_ = KMeansGPU(self.cfg.dataset.n_clusters).fit(embedding)
            return embedding, labels_, clustering_centers_
        if method == 'kmeans_cpu' or self.device == 'cpu':
            embedding = embedding.cpu().numpy()
            kmeans = KMeans(self.cfg.dataset.n_clusters, n_init=20)
            kmeans.fit_predict(embedding)
            labels_ = kmeans.labels_
            clustering_centers_ = kmeans.cluster_centers_
            labels_, clustering_centers_ = torch.from_numpy(labels_), torch.from_numpy(clustering_centers_)
            return torch.from_numpy(embedding), labels_, clustering_centers_

    def evaluate(self, data: Data):
        embedding, labels, clustering_centers = self.clustering(data)
        ground_truth = data.y.numpy()
        metric = DGCMetric(ground_truth, labels.numpy(), embedding, data.edge_index)
        metric.evaluate_one_epoch(self.logger, acc=True, nmi=True, ari=True, f1=True, hom=True, com=True, pur=True, sc=True, gre=True)
