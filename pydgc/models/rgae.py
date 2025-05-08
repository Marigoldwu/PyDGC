# -*- coding: utf-8 -*-
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from typing import Tuple, List, Any

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from torch import Tensor

from . import DGCModel
from ..metrics import DGCMetric
from ..utils import Logger, Data, validate_and_create_path
from yacs.config import CfgNode as CN


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def target_distribution(q):
    p = torch.nn.functional.one_hot(torch.argmax(q, dim=1), q.shape[1]).to(dtype=torch.float32)
    return p


def random_uniform_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return nn.Parameter(initial)


def q_mat(X, centers, alpha=1.0):
    if X.size == 0:
        q = np.array([])
    else:
        q = 1.0 / (1.0 + (np.sum(np.square(np.expand_dims(X, 1) - centers), axis=2) / alpha))
        q = q ** ((alpha + 1.0) / 2.0)
        q = np.transpose(np.transpose(q) / np.sum(q, axis=1))
    return q


def generate_unconflicted_data_index(emb, centers_emb, beta1, beta2):
    unconf_indices = []
    conf_indices = []
    q = q_mat(emb, centers_emb, alpha=1.0)
    confidence1 = np.zeros((q.shape[0],))
    confidence2 = np.zeros((q.shape[0],))
    a = np.argsort(q, axis=1)
    for i in range(q.shape[0]):
        confidence1[i] = q[i, a[i, -1]]
        confidence2[i] = q[i, a[i, -2]]
        if (confidence1[i]) > beta1 and (confidence1[i] - confidence2[i]) > beta2:
            unconf_indices.append(i)
        else:
            conf_indices.append(i)
    unconf_indices = np.asarray(unconf_indices, dtype=int)
    conf_indices = np.asarray(conf_indices, dtype=int)
    return unconf_indices, conf_indices


class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, activation=F.relu, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = random_uniform_init(input_dim, output_dim)
        self.activation = activation

    def forward(self, inputs, adj):
        x = inputs
        x = torch.mm(x, self.weight)
        x = torch.mm(adj, x)
        outputs = self.activation(x)
        return outputs


class ClusterAssignment(nn.Module):
    def __init__(self, cluster_number, embedding_dimension, alpha, cluster_centers=None):
        super(ClusterAssignment, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(self.cluster_number, self.embedding_dimension, dtype=torch.float)
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = nn.Parameter(initial_cluster_centers)

    def forward(self, inputs):
        norm_squared = torch.sum((inputs.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)


class RGAE(DGCModel):
    def __init__(self, logger: Logger, cfg: CN):
        super(RGAE, self).__init__(logger, cfg)
        dims = cfg.model.dims
        dims.insert(0, cfg.dataset.num_features)
        # dims, activation, alpha
        if cfg.model.activation == "ReLU":
            activation = F.relu
        elif cfg.model.activation == "Sigmoid":
            activation = F.sigmoid
        else:
            activation = F.tanh
        self.gcn_1 = GraphConvSparse(dims[0], dims[1], activation).to(self.device)
        self.gcn_2 = GraphConvSparse(dims[1], dims[-1], activation=lambda x: x).to(self.device)
        self.assignment = ClusterAssignment(cfg.dataset.n_clusters, dims[-1], cfg.model.alpha).to(self.device)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean').to(self.device)

        self.loss_curve = []
        self.pretrain_loss_curve = []

    def reset_parameters(self):
        pass

    def forward(self, data) -> Any:
        x_features, adj = data.x.to(self.device), data.adj_norm.to(self.device)
        hidden = self.gcn_1(x_features, adj)
        embedding = self.gcn_2(hidden, adj)
        hat_adj = torch.sigmoid(torch.matmul(embedding, embedding.t()))
        return hat_adj, embedding

    def pretrain(self, data: Data, cfg: CN = None, flag: str = "PRETRAIN RGAE"):
        # pretrain: dir, optimizer, lr, max_epoch
        if cfg is None:
            cfg = self.cfg.train.pretrain
        self.logger.flag(flag)
        if cfg.optimizer == "Adam":
            opti = torch.optim.Adam(self.parameters(), lr=float(cfg.lr), weight_decay=0.001)
        elif cfg.optimizer == "SGD":
            opti = torch.optim.SGD(self.parameters(), lr=float(cfg.lr), momentum=0.9)
        else:
            opti = torch.optim.RMSprop(self.parameters(), lr=float(cfg.lr))
        embedding = None
        adj_label = data.adj_label.to(self.device)
        weight_tensor = data.weight_tensor.to(self.device)
        for epoch in range(1, cfg.max_epoch + 1):
            opti.zero_grad()
            hat_adj, embedding = self.forward(data)
            loss = F.cross_entropy(hat_adj.view(-1), adj_label.to_dense().view(-1), weight=weight_tensor)
            self.logger.loss(epoch, loss)
            loss.backward()
            opti.step()
            self.pretrain_loss_curve.append(loss.item())
        km = KMeans(n_clusters=self.cfg.dataset.n_clusters, n_init=20)
        km.fit(embedding.detach().cpu().numpy())
        centers = torch.tensor(km.cluster_centers_, dtype=torch.float, requires_grad=True)
        self.assignment.state_dict()["cluster_centers"].copy_(centers)
        validate_and_create_path(cfg.dir)
        pretrain_file_name = os.path.join(cfg.dir, f'rgae.pth')
        torch.save(self.state_dict(), pretrain_file_name)

    def loss(self, q, p, hat_adj, adj_label, weight_tensor):
        p = p.to(self.device)
        q = q.to(self.device)
        weight_tensor = weight_tensor.to(self.device)
        adj_label = adj_label.to(self.device)
        hat_adj = torch.sigmoid(hat_adj)
        loss_recons = F.cross_entropy(hat_adj.view(-1), adj_label.to_dense().view(-1), weight=weight_tensor)
        loss_clus = self.kl_loss(torch.log(q), p)
        loss = loss_recons + float(self.cfg.train.gamma) * loss_clus
        return loss, loss_recons, loss_clus

    def train_model(self, data: Data, cfg: CN = None, flag: str = "TRAIN RGAE") -> List:
        # optimizer, lr, max_epoch, gamma, beta1, beta2
        if cfg is None:
            cfg = self.cfg.train
        # load pretrained gae model
        pretrain_file_name = os.path.join(cfg.pretrain.dir, f'rgae.pth')
        if not os.path.exists(pretrain_file_name):
            self.pretrain(data, cfg.pretrain, flag='PRETRAIN RGAE')
        self.load_state_dict(torch.load(pretrain_file_name, map_location=self.device))

        if cfg.optimizer == "Adam":
            opti = torch.optim.Adam(self.parameters(), lr=float(cfg.lr), weight_decay=0.01)
        elif cfg.optimizer == "SGD":
            opti = torch.optim.SGD(self.parameters(), lr=float(cfg.lr), momentum=0.9, weight_decay=0.01)
        else:
            opti = torch.optim.RMSprop(self.parameters(), lr=float(cfg.lr), weight_decay=0.01)
        lr_s = torch.optim.lr_scheduler.StepLR(opti, step_size=10, gamma=0.9)
        epoch_stable = 0
        previous_un_conflicted = []
        adj = data.adj
        adj_label = data.adj_label.to(self.device)
        weight_tensor = data.weight_tensor.to(self.device)
        beta1 = float(cfg.beta1)
        beta2 = float(cfg.beta2)
        for epoch in range(cfg.max_epoch):
            opti.zero_grad()
            hat_adj, embedding = self.forward(data)
            q = self.assignment(embedding)
            if epoch % 15 == 0:
                p = target_distribution(q.detach())
            if epoch % 15 == 0:
                un_conflicted_ind, conflicted_ind = generate_unconflicted_data_index(embedding.detach().cpu().numpy(),
                                                                                    self.assignment.cluster_centers.detach().cpu().numpy(),
                                                                                    beta1, beta2)
                if epoch == 0:
                    adj, adj_label, weight_tensor = self.update_graph(adj, embedding, un_conflicted_ind)
            if len(previous_un_conflicted) < len(un_conflicted_ind):
                p_un_conf = p[un_conflicted_ind]
                q_un_conf = q[un_conflicted_ind]
                previous_un_conflicted = un_conflicted_ind
            else:
                epoch_stable += 1
                p_un_conf = p[previous_un_conflicted]
                q_un_conf = q[previous_un_conflicted]
            if epoch_stable >= 15:
                epoch_stable = 0
                beta1 = beta1 * 0.95
                beta2 = beta2 * 0.85
            if epoch % 20 == 0 and epoch <= 120:
                adj, adj_label, weight_tensor = self.update_graph(adj, embedding, un_conflicted_ind)
            loss, _, _ = self.loss(q_un_conf, p_un_conf, hat_adj, adj_label, weight_tensor)

            loss.backward()
            opti.step()
            lr_s.step()
            self.loss_curve.append(loss.item())
            self.logger.loss(epoch, loss)
        return self.loss_curve

    def generate_centers(self, emb_unconf, y_pred):
        nn_ = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(emb_unconf.detach().cpu().numpy())
        _, indices = nn_.kneighbors(self.assignment.cluster_centers.detach().cpu().numpy())
        return indices[y_pred]

    def update_graph(self, adj, emb, un_conf_indices):
        y_pred = self.predict(emb)
        emb_un_conf = emb[un_conf_indices]
        adj = adj.tolil()
        idx = un_conf_indices[self.generate_centers(emb_un_conf, y_pred)]
        for i, k in enumerate(un_conf_indices):
            adj_k = adj[k].tocsr().indices
            if not (np.isin(idx[i], adj_k)) and (y_pred[k] == y_pred[idx[i]]):
                adj[k, idx[i]] = 1
            for j in adj_k:
                if np.isin(j, un_conf_indices) and (y_pred[k] != y_pred[j]):
                    adj[k, j] = 0
        adj = adj.tocsr()
        adj_label = adj + sp.eye(adj.shape[0])
        adj_label = sparse_to_tuple(adj_label)
        adj_label = torch.sparse_coo_tensor(torch.LongTensor(adj_label[0].T), torch.FloatTensor(adj_label[1]),
                                            torch.Size(adj_label[2]))
        weight_mask = adj_label.to_dense().view(-1) == 1
        weight_tensor = torch.ones(weight_mask.size(0))
        pos_weight_orig = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        weight_tensor[weight_mask] = pos_weight_orig
        return adj, adj_label, weight_tensor

    def predict(self, emb):
        with torch.no_grad():
            q = self.assignment(emb)
            out = np.argmax(q.detach().cpu().numpy(), axis=1)
        return out

    def get_embedding(self, data) -> Tensor:
        with torch.no_grad():
            hat_adj, embedding = self.forward(data)
            return embedding.detach()

    def clustering(self, data) -> Tuple[Tensor, Tensor, Tensor]:
        embedding = self.get_embedding(data)
        labels_ = self.predict(embedding)
        clustering_centers = self.assignment.cluster_centers.data
        return embedding, labels_, clustering_centers

    def evaluate(self, data):
        embedding, labels, clustering_centers = self.clustering(data)
        ground_truth = data.y.numpy()
        metric = DGCMetric(ground_truth, labels.numpy(), embedding, data.edge_index)
        metric.evaluate_one_epoch(self.logger, acc=True, nmi=True, ari=True, f1=True, hom=True, com=True, pur=True)
