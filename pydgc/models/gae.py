# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch import Tensor
from typing import Tuple

from ..metrics import DGCMetric
from ..utils import Logger
from .dgc_model import DGCModel
from torch_geometric.data import Data
from yacs.config import CfgNode as CN
from ..clusterings import KMeansGPU
from torch_geometric.utils import to_dense_adj
from ..modules import GNNEncoder, InnerProductDecoder


class GAE(DGCModel):
    def __init__(self, logger: Logger, cfg: CN):
        super(GAE, self).__init__(logger, cfg)
        self.device = torch.device(cfg.device)
        dims = cfg.model.dims.copy()
        dims.insert(0, self.cfg.dataset.num_features)
        self.encoder = GNNEncoder(dims=dims, layer=self.cfg.model.gnn_type.lower(), act=self.cfg.model.act, act_last=self.cfg.model.act_last).to(self.device)
        self.decoder = InnerProductDecoder().to(self.device)
        self.loss_curve = []
        self.best_embedding = None
        self.best_predicted_labels = None
        self.best_results = {'ACC': -1}
        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()

    def forward(self, data):
        x = data.x.to(self.device).float()
        edge_index = data.edge_index.to(self.device)
        embedding = F.normalize(self.encoder(x, edge_index), p=2, dim=1)
        hat_adj = self.decoder(embedding)
        return hat_adj, embedding

    def loss(self, edge_index, hat_adj: Tensor) -> Tensor:
        dense_adj = to_dense_adj(edge_index)[0].to(self.device)
        loss = F.cross_entropy(hat_adj.view(-1), dense_adj.view(-1))
        return loss

    def train_model(self, data: Data, cfg: CN = None, flag: str = "TRAIN GAE"):
        flag += f'-{self.cfg.model.gnn_type.upper()}'
        # when gae is trained in pre-training mode, cfg.pretrain must be input as parameter
        if cfg is None:
            cfg = self.cfg.train
        self.logger.flag(flag)

        optimizer = torch.optim.Adam(self.parameters(), lr=float(cfg.lr))
        for epoch in range(1, cfg.max_epoch+1):
            self.train()
            optimizer.zero_grad()
            hat_adj, _ = self.forward(data)
            loss = self.loss(data.edge_index, hat_adj)
            loss.backward()
            optimizer.step()
            self.loss_curve.append(loss.item())
            self.logger.loss(epoch, loss)
            if self.cfg.evaluate.each:
                embedding, predicted_labels, results = self.evaluate(data)
                if results['ACC'] > self.best_results['ACC']:
                    self.best_embedding = embedding
                    self.best_predicted_labels = predicted_labels
                    self.best_results = results
        if not self.cfg.evaluate.each:
            embedding, predicted_labels, results = self.evaluate(data)
            return self.loss_curve, embedding, predicted_labels, results
        return self.loss_curve, self.best_embedding, self.best_predicted_labels, self.best_results

    def get_embedding(self, data: Data) -> Tensor:
        x = data.x.to(self.device).float()
        edge_index = data.edge_index.to(self.device)
        with torch.no_grad():
            self.eval()
            embedding = F.normalize(self.encoder(x, edge_index), p=2, dim=1)
            return embedding

    def clustering(self, data: Data, method: str = 'kmeans_gpu') -> Tuple[Tensor, Tensor, Tensor]:
        embedding = self.get_embedding(data)
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
        embedding, predicted_labels, clustering_centers = self.clustering(data)
        ground_truth = data.y.numpy()
        metric = DGCMetric(ground_truth, predicted_labels.numpy(), embedding, data.edge_index)
        results = metric.evaluate_one_epoch(self.logger, acc=True, nmi=True, ari=True, f1=True, hom=True, com=True, pur=True)
        return embedding, predicted_labels, results
