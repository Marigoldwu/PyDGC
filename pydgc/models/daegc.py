# -*- coding: utf-8 -*-
import os

import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

from torch import Tensor
from torch_geometric.data import Data

from ..utils import Logger, validate_and_create_path
from ..metrics import DGCMetric
from ..models import DGCModel
from ..clusterings import KMeansGPU
from ..modules import GATMEncoder, InnerProductDecoder, SSCLayer
from typing import Tuple, List
from yacs.config import CfgNode as CN


class GATE(DGCModel):
    def __init__(self, logger: Logger, cfg: CN):
        super(GATE, self).__init__(logger, cfg)
        self.device = torch.device(cfg.device)
        dims = cfg.model.dims.copy()
        dims.insert(0, self.cfg.dataset.num_features)
        self.gat_encoder = GATMEncoder(dims=dims).to(self.device)
        self.decoder = InnerProductDecoder().to(self.device)
        self.loss_curve = []
        self.results = {}
        self.reset_parameters()

    def reset_parameters(self):
        self.gat_encoder.reset_parameters()
        self.decoder.reset_parameters()

    def forward(self, data):
        x = data.x.to(self.device)
        adj = data.adj.to(self.device)
        M = data.M.to(self.device)

        embedding = F.normalize(self.gat_encoder(x, adj, M), p=2, dim=1)
        hat_adj = self.decoder(embedding)
        return hat_adj, embedding

    def loss(self, adj_label: Tensor, hat_adj: Tensor) -> Tensor:
        adj_label = adj_label.to(self.device)
        loss = F.cross_entropy(hat_adj.view(-1), adj_label.view(-1))
        return loss

    def train_model(self, data: Data, cfg: CN = None, flag: str = "TRAIN GATE") -> List:
        # when gate is trained in pre-training mode, cfg.pretrain must be input as parameter
        if cfg is None:
            cfg = self.cfg.train
        self.logger.flag(flag)

        optimizer = torch.optim.Adam(self.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
        for epoch in range(1, cfg.max_epoch + 1):
            self.train()
            optimizer.zero_grad()
            hat_adj, _ = self.forward(data)
            loss = self.loss(data.adj_label, hat_adj)
            loss.backward()
            optimizer.step()
            self.loss_curve.append(loss.item())
            self.logger.loss(epoch, loss)
            if self.cfg.evaluate.each:
                self.evaluate(data)
        return self.loss_curve

    def get_embedding(self, data) -> Tensor:
        x = data.x.to(self.device)
        adj = data.adj.to(self.device)
        M = data.M.to(self.device)

        with torch.no_grad():
            self.eval()
            embedding = F.normalize(self.gat_encoder(x, adj, M), p=2, dim=1)
            return embedding

    def clustering(self, data, method='kmeans_gpu') -> Tuple[Tensor, Tensor, Tensor]:
        embedding = self.get_embedding(data)
        if method == 'kmeans_gpu':
            labels_, clustering_centers_ = KMeansGPU(self.cfg.dataset.n_clusters).fit(embedding)
            return embedding, labels_, clustering_centers_
        if method == 'kmeans_cpu' or self.device == 'cpu':
            embedding = embedding.cpu().numpy()
            kmeans = KMeans(self.cfg.dataset.n_clusters, n_init=20)
            kmeans.fit_predict(embedding)
            clustering_centers_ = kmeans.cluster_centers_
            labels_ = kmeans.labels_
            labels_, clustering_centers_ = torch.from_numpy(labels_), torch.from_numpy(clustering_centers_)
            return torch.from_numpy(embedding), labels_, clustering_centers_

    def evaluate(self, data: Data):
        embedding, labels, clustering_centers = self.clustering(data)
        ground_truth = data.y.numpy()
        metric = DGCMetric(ground_truth, labels.numpy(), embedding, data.edge_index)
        metric.evaluate_one_epoch(self.logger, acc=True, nmi=True, ari=True, f1=True, hom=True, com=True, pur=True)


class DAEGC(DGCModel):

    def __init__(self, logger: Logger, cfg: CN):
        super(DAEGC, self).__init__(logger, cfg)
        self.device = torch.device(cfg.device)
        self.gate = GATE(logger, cfg).to(self.device)
        self.ssc = SSCLayer(in_channels=self.cfg.model.dims[-1], out_channels=self.cfg.dataset.n_clusters, method='kl_div').to(self.device)
        self.loss_curve = []
        self.pretrain_loss_curve = []
        self.best_embedding = None
        self.best_predicted_labels = None
        self.best_results = {'ACC': -1}
        self.reset_parameters()

    def reset_parameters(self):
        self.gate.reset_parameters()
        self.ssc.reset_parameters()

    def forward(self, data):
        hat_adj, embedding = self.gate(data)
        q = self.ssc(embedding)
        return hat_adj, q

    def loss(self, adj_label: Tensor, hat_adj: Tensor, q: Tensor) -> Tensor:
        reconstruct_loss = self.gate.loss(adj_label, hat_adj)
        ssc_loss = self.ssc.loss(q, method='kl_div')
        loss_total = reconstruct_loss + float(self.cfg.train.gamma) * ssc_loss
        return loss_total

    def pretrain(self, data: Data, cfg: CN = None, flag: str = "PRETRAIN GATE"):
        if cfg is None:
            cfg = self.cfg.train.pretrain
        self.pretrain_loss_curve = self.gate.train_model(data, cfg, flag)
        validate_and_create_path(cfg.dir)
        pretrain_file_name = os.path.join(cfg.dir, f'gate.pth')
        torch.save(self.gate.state_dict(), pretrain_file_name)

    def train_model(self, data: Data, cfg: CN = None, flag: str = "TRAIN DAEGC"):
        if cfg is None:
            cfg = self.cfg.train
        # load pretrained gate model
        pretrain_file_name = os.path.join(cfg.pretrain.dir, f'gate.pth')
        if not os.path.exists(pretrain_file_name):
            self.pretrain(data, cfg.pretrain, flag='PRETRAIN GATE')
        self.gate.load_state_dict(torch.load(pretrain_file_name, map_location='cpu'))

        optimizer = torch.optim.Adam(self.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

        # initialize ssc layer
        _, _, cluster_centers = self.gate.clustering(data)
        self.ssc.cluster_centers.data = cluster_centers.to(self.device)
        self.gate.evaluate(data)

        self.logger.flag(flag)
        # train
        for epoch in range(1, cfg.max_epoch + 1):
            self.train()
            optimizer.zero_grad()

            hat_adj, q = self.forward(data)
            loss = self.loss(data.adj, hat_adj, q)
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

    def get_embedding(self, data) -> Tensor:
        with torch.no_grad():
            self.eval()
            return self.gate.get_embedding(data)

    def clustering(self, data) -> Tuple[Tensor, Tensor, Tensor]:
        embedding = self.get_embedding(data)
        labels_ = torch.from_numpy(self.ssc.get_q(embedding).detach().cpu().numpy().argmax(axis=1))
        clustering_centers = self.ssc.cluster_centers.data
        return embedding, labels_, clustering_centers

    def evaluate(self, data: Data):
        embedding, predicted_labels, clustering_centers = self.clustering(data)
        ground_truth = data.y.numpy()
        metric = DGCMetric(ground_truth, predicted_labels.numpy(), embedding, data.edge_index)
        results = metric.evaluate_one_epoch(self.logger, acc=True, nmi=True, ari=True, f1=True, hom=True, com=True, pur=True)
        return embedding, predicted_labels, results
