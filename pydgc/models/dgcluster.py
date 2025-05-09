# -*- coding: utf-8 -*-
import random
from typing import Tuple, List, Any

from sklearn.cluster import Birch
from torch import Tensor
from torch.optim import lr_scheduler
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv

from . import DGCModel
from yacs.config import CfgNode as CN
import numpy as np

from ..metrics import DGCMetric
from ..utils import Logger
import torch
import torch.nn as nn
import torch.nn.functional as F


def convert_scipy_torch_sp(sp_adj):
    sp_adj = sp_adj.tocoo()
    indices = torch.tensor(np.vstack((sp_adj.row, sp_adj.col)))
    sp_adj = torch.sparse_coo_tensor(indices, torch.tensor(sp_adj.data), size=sp_adj.shape)
    return sp_adj


def aux_objective(output, s, oh_labels):
    sample_size = len(s)

    out = output[s, :].float()

    C = oh_labels[s, :].float()

    X = C.sum(dim=0)
    X = X ** 2
    X = X.sum()

    Y = torch.matmul(torch.t(out), C)
    Y = torch.matmul(Y, torch.t(Y))
    Y = torch.trace(Y)

    t1 = torch.matmul(torch.t(C), C)
    t1 = torch.matmul(t1, t1)
    t1 = torch.trace(t1)

    t2 = torch.matmul(torch.t(out), out)
    t2 = torch.matmul(t2, t2)
    t2 = torch.trace(t2)

    t3 = torch.matmul(torch.t(out), C)
    t3 = torch.matmul(t3, torch.t(t3))
    t3 = torch.trace(t3)

    aux_objective_loss = 1 / (sample_size ** 2) * (t1 + t2 - 2 * t3)

    return aux_objective_loss


def regularization(output, s):
    out = output[s, :]
    ss = out.sum(dim=0)
    ss = ss ** 2
    ss = ss.sum()
    avg_sim = 1 / (len(s) ** 2) * ss

    return avg_sim ** 2


class DGCLUSTER(DGCModel):

    def __init__(self, logger: Logger, cfg: CN):
        super(DGCLUSTER, self).__init__(logger, cfg)
        dims = cfg.model.dims.copy()
        dims.insert(0, cfg.dataset.num_features)

        if cfg.model.gnn_type == 'gcn':
            self.conv1 = GCNConv(dims[0], dims[1])
            self.conv2 = GCNConv(dims[1], dims[2])
            self.conv3 = GCNConv(dims[2], dims[-1])
        elif cfg.model.gnn_type == 'gat':
            self.conv1 = GATConv(dims[0], dims[1])
            self.conv2 = GATConv(dims[1], dims[2])
            self.conv3 = GATConv(dims[2], dims[-1])
        elif cfg.model.gnn_type == 'gin':
            self.conv1 = GINConv(nn.Linear(dims[0], dims[1]))
            self.conv2 = GINConv(nn.Linear(dims[1], dims[2]))
            self.conv3 = GINConv(nn.Linear(dims[2], dims[-1]))
        else:
            self.conv1 = SAGEConv(dims[0], dims[1])
            self.conv2 = SAGEConv(dims[1], dims[2])
            self.conv3 = SAGEConv(dims[2], dims[-1])

        self.conv1.to(self.device)
        self.conv2.to(self.device)
        self.conv3.to(self.device)

        self.loss_curve = []

    def reset_parameters(self):
        pass

    def forward(self, data:  Data) -> Any:
        x, edge_index = data.x.to(self.device), data.edge_index.to(self.device)

        x = self.conv1(x, edge_index)
        x = F.selu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.selu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv3(x, edge_index)

        x = x / (x.sum())
        x = (F.tanh(x)) ** 2
        x = F.normalize(x)

        return x

    def loss(self, output, data, lam, alp) -> Tensor:
        num_nodes = self.cfg.dataset.num_nodes
        num_edges = self.cfg.dataset.num_edges
        sparse_adj = data.sparse_adj
        degree = data.degree

        sample_size = int(1 * num_nodes)
        s = random.sample(range(0, num_nodes), sample_size)

        s_output = output[s, :]

        s_adj = sparse_adj[s, :][:, s]
        s_adj = convert_scipy_torch_sp(s_adj)
        s_degree = degree[s]

        x = torch.matmul(torch.t(s_output).double(), s_adj.double().to(self.device))
        x = torch.matmul(x, s_output.double())
        x = torch.trace(x)

        y = torch.matmul(torch.t(s_output).double(), s_degree.double().to(self.device))
        y = (y ** 2).sum()
        y = y / (2 * num_edges)

        # scaling=1
        scaling = num_nodes ** 2 / (sample_size ** 2)

        m_loss = -((x - y) / (2 * num_edges)) * scaling

        aux_loss = lam * aux_objective(output, s, data.oh_labels)

        reg_loss = alp * regularization(output, s)

        loss = m_loss + aux_loss + reg_loss

        return loss

    def train_model(self, data: Data, cfg: CN = None, flag: str = "TRAIN DGCLUSTER") -> List:
        if cfg is None:
            cfg = self.cfg.train
        self.logger.flag(flag)

        optimizer = torch.optim.Adam(self.parameters(), lr=float(cfg.lr), betas=(0.9, 0.999), weight_decay=0.001, amsgrad=True)
        scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=cfg.max_epoch)

        self.train()
        for epoch in range(cfg.max_epoch):
            optimizer.zero_grad()
            out = self.forward(data)

            loss = self.loss(out, data, cfg.lam, cfg.alp)
            loss.backward()
            self.loss_curve.append(loss.item())
            self.logger.loss(epoch, loss)

            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.1)
            optimizer.step()
            scheduler.step()
            if self.cfg.evaluate.each:
                self.evaluate(data)
        return self.loss_curve

    def get_embedding(self, data: Data) -> Tensor:
        with torch.no_grad():
            embedding = self.forward(data)
        return embedding.detach()

    def clustering(self, data: Data) -> Tuple[Tensor, Tensor, Tensor]:
        embedding = self.get_embedding(data)
        birch = Birch(n_clusters=self.cfg.dataset.n_clusters, threshold=0.5)
        labels = torch.from_numpy(birch.fit_predict(embedding.cpu().numpy()))
        return embedding, labels, birch.subcluster_centers_

    def evaluate(self, data: Data):
        embedding, labels, clustering_centers = self.clustering(data)
        ground_truth = data.y.cpu().numpy()
        metric = DGCMetric(ground_truth, labels.numpy(), embedding, data.edge_index)
        metric.evaluate_one_epoch(self.logger, acc=True, nmi=True, ari=True, f1=True, hom=True, com=True, pur=True,
                                  sc=True, gre=True)


