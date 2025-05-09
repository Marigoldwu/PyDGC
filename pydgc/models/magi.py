# -*- coding: utf-8 -*-
from typing import Tuple, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..clusterings import kmeans
from sklearn.cluster import SpectralClustering, KMeans
from torch import Tensor
from torch_geometric.data import Data

from torch_geometric.nn import GCNConv

from . import DGCModel

from yacs.config import CfgNode as CN

from ..metrics import DGCMetric
from ..utils import Logger
import numpy as np


class Encoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels, base_model=GCNConv, dropout: float = 0.5, ns: float = 0.5):
        super(Encoder, self).__init__()
        self.base_model = base_model
        self.dropout = dropout
        self.k = len(hidden_channels)
        self.ns = ns

        self.convs = nn.ModuleList()
        self.convs.extend([base_model(in_channels, hidden_channels[0])])

        for i in range(1, self.k):
            self.convs.extend(
                [base_model(hidden_channels[i - 1], hidden_channels[i])])

        self.reset_parameters()

    def reset_parameters(self):
        """初始化模型参数"""
        for i in range(self.k):
            self.convs[i].reset_parameters()

    def forward(self, x: torch.Tensor, edge_index=None, adjs=None, dropout=True):
        if not adjs:
            for i in range(self.k):
                if dropout:
                    x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.convs[i](x, edge_index)
                x = F.leaky_relu(x, self.ns)
        else:
            for i, (edge_index, _, size) in enumerate(adjs):
                if dropout:
                    x = F.dropout(x, p=self.dropout, training=self.training)
                x_target = x[:size[1]]  # Target nodes are always placed first.
                x = self.convs[i]((x, x_target), edge_index)
                x = F.leaky_relu(x, self.ns)
        return x


class Loss(nn.Module):
    def __init__(self, temperature=0.07, scale_by_temperature=True, scale_by_weight=False):
        super(Loss, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature
        self.scale_by_weight = scale_by_weight

    def forward(self, out, mask):
        device = (torch.device('cuda') if out.is_cuda else torch.device('cpu'))

        row, col, val = mask.storage.row(), mask.storage.col(), mask.storage.value()
        row, col, val = row.to(device), col.to(device), val.to(device)
        batch_size = out.shape[0]

        # compute logits
        dot = torch.matmul(out, out.T)
        dot = torch.div(dot, self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(dot, dim=1, keepdim=True)
        dot = dot - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones(batch_size, batch_size).to(device),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )

        exp_logits = torch.exp(dot) * logits_mask
        log_probs = dot - torch.log(exp_logits.sum(1, keepdim=True))

        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")

        labels = row.view(row.shape[0], 1)
        unique_labels, labels_count = labels.unique(dim=0, return_counts=True)
        log_probs = log_probs[row, col]

        log_probs = log_probs.view(-1, 1)
        loss = torch.zeros_like(unique_labels, dtype=torch.float).to(device)
        loss.scatter_add_(0, labels, log_probs)
        loss = -1 * loss / labels_count.float().unsqueeze(1)

        if self.scale_by_temperature:
            loss *= self.temperature
        loss = loss.mean()
        return loss


def clustering(feature, n_clusters, true_labels, kmeans_device='cpu', batch_size=100000, tol=1e-4,
               device=torch.device('cuda:0'), spectral_clustering=False):
    if spectral_clustering:
        if isinstance(feature, torch.Tensor):
            feature = feature.numpy()
        print("spectral clustering on cpu...")
        Cluster = SpectralClustering(
            n_clusters=n_clusters, affinity='precomputed', random_state=0)
        f_adj = np.matmul(feature, np.transpose(feature))
        predict_labels = Cluster.fit_predict(f_adj)
    else:
        if kmeans_device == 'cuda':
            if isinstance(feature, np.ndarray):
                feature = torch.tensor(feature)
            print("kmeans on gpu...")
            predict_labels, _ = kmeans(
                X=feature, num_clusters=n_clusters, batch_size=batch_size, tol=tol, device=device)
            predict_labels = predict_labels.numpy()
        else:
            if isinstance(feature, torch.Tensor):
                feature = feature.numpy()
            print("kmeans on cpu...")
            Cluster = KMeans(n_clusters=n_clusters, max_iter=10000, n_init=20)
            predict_labels = Cluster.fit_predict(feature)
    return torch.from_numpy(predict_labels), None


def scale(z: torch.Tensor):
    zmax = z.max(dim=1, keepdim=True)[0]
    zmin = z.min(dim=1, keepdim=True)[0]
    z_std = (z - zmin) / ((zmax - zmin) + 1e-20)
    z_scaled = z_std
    return z_scaled


class MAGI(DGCModel):

    def __init__(self, logger: Logger, cfg: CN):
        super(MAGI, self).__init__(logger, cfg)
        encoder_dims = cfg.model.dims.encoder.copy()
        projection_dims = cfg.model.dims.projection
        encoder_dims.insert(0, cfg.dataset.num_features)
        self.encoder = Encoder(encoder_dims[0], encoder_dims[1:], base_model=GCNConv,
                               dropout=cfg.model.dropout, ns=cfg.model.ns).to(self.device)
        self.tau = cfg.model.tau
        self.in_channels = encoder_dims[-1]
        self.project_hidden = projection_dims if projection_dims != "" else None
        self.activation = nn.PReLU
        self.Loss = Loss(temperature=self.tau)

        self.project = None
        if self.project_hidden is not None:
            self.project = nn.ModuleList()
            self.activations = nn.ModuleList()
            self.project.extend(
                [nn.Linear(self.in_channels, self.project_hidden[0])])
            self.activations.extend([nn.PReLU(projection_dims[0])])
            for i in range(1, len(self.project_hidden)):
                self.project.extend(
                    [nn.Linear(self.project_hidden[i - 1], self.project_hidden[i])])
                self.activations.extend([nn.PReLU(projection_dims[i])])
            self.project.to(self.device)

        self.loss_curve = []

        self.best_embedding = None
        self.best_predicted_labels = None
        self.best_results = {'ACC': -1}

    def reset_parameters(self):
        pass

    def forward(self, data) -> Any:
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        x = self.encoder(x, edge_index)
        if self.project is not None:
            for i in range(len(self.project_hidden)):
                x = self.project[i](x)
                x = self.activations[i](x)
        return x

    def loss(self, *args, **kwargs) -> Tensor:
        pass

    def train_model(self, data: Data, cfg: CN = None, flag: str = "TRAIN MAGI"):
        if cfg is None:
            cfg = self.cfg.train
        optimizer = torch.optim.Adam(self.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
        mask = data.mask.to(self.device)
        # train
        for epoch in range(1, cfg.max_epoch + 1):
            self.train()
            optimizer.zero_grad()
            out = self.forward(data)
            out = scale(out)
            out = F.normalize(out, p=2, dim=1)
            loss = self.Loss(out, mask)
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
        # eval
        with torch.no_grad():
            self.eval()
            out = self.forward(data)
            out = scale(out)
            embedding = F.normalize(out, p=2, dim=1)
        return embedding.detach()

    def clustering(self, data) -> Tuple[Tensor, Tensor, Any]:
        embedding = self.get_embedding(data)
        labels, clustering_centers = clustering(embedding.cpu().numpy(), self.cfg.dataset.n_clusters, data.y,
                                                spectral_clustering=True)
        return embedding, labels, clustering_centers

    def evaluate(self, data):
        embedding, predicted_labels, clustering_centers = self.clustering(data)
        ground_truth = data.y.numpy()
        metric = DGCMetric(ground_truth, predicted_labels.numpy(), embedding, data.edge_index)
        results = metric.evaluate_one_epoch(self.logger, acc=True, nmi=True, ari=True, f1=True, hom=True, com=True, pur=True, sc=True, gre=True)
        return embedding, predicted_labels, results
