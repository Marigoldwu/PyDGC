# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Data

from ..clusterings import KMeansGPU
from ..metrics import DGCMetric
from ..utils import Logger
from typing import Tuple, Any

from torch import Tensor
from yacs.config import CfgNode as CN

from . import DGCModel


def comprehensive_similarity(Z1, Z2, E1, E2, alpha):
    Z1_Z2 = torch.cat([torch.cat([Z1 @ Z1.T, Z1 @ Z2.T], dim=1),
                       torch.cat([Z2 @ Z1.T, Z2 @ Z2.T], dim=1)], dim=0)

    E1_E2 = torch.cat([torch.cat([E1 @ E1.T, E1 @ E2.T], dim=1),
                       torch.cat([E2 @ E1.T, E2 @ E2.T], dim=1)], dim=0)

    S = alpha * Z1_Z2 + (1 - alpha) * E1_E2
    return S


def hard_sample_aware_infoNCE(S, M, pos_neg_weight, pos_weight, node_num):
    pos_neg = M * torch.exp(S * pos_neg_weight)
    pos = torch.cat([torch.diag(S, node_num), torch.diag(S, -node_num)], dim=0)
    pos = torch.exp(pos * pos_weight)
    neg = (torch.sum(pos_neg, dim=1) - pos)
    infoNEC = (-torch.log(pos / (pos + neg))).sum() / (2 * node_num)
    return infoNEC


def square_euclid_distance(Z, center):
    ZZ = (Z * Z).sum(-1).reshape(-1, 1).repeat(1, center.shape[0])
    CC = (center * center).sum(-1).reshape(1, -1).repeat(Z.shape[0], 1)
    ZZ_CC = ZZ + CC
    ZC = Z @ center.T
    distance = ZZ_CC - 2 * ZC
    return distance


def phi(embedding, cluster_num):
    labels_, clustering_centers_ = KMeansGPU(cluster_num).fit(embedding)
    return labels_, clustering_centers_


class HSAN(DGCModel):
    def __init__(self, logger: Logger, cfg: CN):
        super(HSAN, self).__init__(logger, cfg)
        self.device = torch.device(cfg.device)
        input_dim = cfg.dataset.num_features if cfg.model.dims.input_dim == -1 else cfg.model.dims.input_dim
        hidden_dim = cfg.model.dims.hidden_dim
        n_num = cfg.dataset.num_nodes
        self.AE1 = nn.Linear(input_dim, hidden_dim).to(self.device)
        self.AE2 = nn.Linear(input_dim, hidden_dim).to(self.device)

        self.SE1 = nn.Linear(n_num, hidden_dim).to(self.device)
        self.SE2 = nn.Linear(n_num, hidden_dim).to(self.device)

        self.alpha = nn.Parameter(torch.Tensor(1, ))
        self.alpha.data = torch.tensor(0.99999).to(self.device)

        self.pos_weight = torch.ones(n_num * 2).to(self.device)
        self.pos_neg_weight = torch.ones([n_num * 2, n_num * 2]).to(self.device)

        if self.cfg.model.act == "ident":
            self.activate = lambda x: x
        if self.cfg.model.act == "sigmoid":
            self.activate = nn.Sigmoid()

        self.loss_curve = []
        self.best_embedding = None
        self.best_predicted_labels = None
        self.best_results = {'ACC': -1}

    def reset_parameters(self):
        pass

    def forward(self, data) -> Any:
        x = data.x.to(self.device)
        A = data.adj.to(self.device)
        Z1 = self.activate(self.AE1(x))
        Z2 = self.activate(self.AE2(x))

        Z1 = F.normalize(Z1, dim=1, p=2)
        Z2 = F.normalize(Z2, dim=1, p=2)

        E1 = F.normalize(self.SE1(A), dim=1, p=2)
        E2 = F.normalize(self.SE2(A), dim=1, p=2)

        return Z1, Z2, E1, E2

    def high_confidence(self, Z, center):
        distance_norm = torch.min(F.softmax(square_euclid_distance(Z, center), dim=1), dim=1).values
        value, _ = torch.topk(distance_norm, int(Z.shape[0] * (1 - self.cfg.train.tau)))
        index = torch.where(distance_norm <= value[-1],
                            torch.ones_like(distance_norm), torch.zeros_like(distance_norm))

        high_conf_index_v1 = torch.nonzero(index).reshape(-1, )
        high_conf_index_v2 = high_conf_index_v1 + Z.shape[0]
        H = torch.cat([high_conf_index_v1, high_conf_index_v2], dim=0)
        H_mat = np.ix_(H.cpu(), H.cpu())
        return H, H_mat

    def pseudo_matrix(self, P, S, node_num):
        P = P.detach().clone()
        P = torch.cat([P, P], dim=0)
        Q = (P == P.unsqueeze(1)).float().to(self.device)
        S_norm = (S - S.min()) / (S.max() - S.min())
        M_mat = torch.abs(Q - S_norm) ** self.cfg.train.beta
        M = torch.cat([torch.diag(M_mat, node_num), torch.diag(M_mat, -node_num)], dim=0)
        return M, M_mat

    def loss(self, *args, **kwargs) -> Tensor:
        pass

    def train_model(self, data: Data, cfg: CN = None, flag: str = "TRAIN HSAN"):
        if cfg is None:
            cfg = self.cfg.train
        node_num = self.cfg.dataset.num_nodes
        # adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=float(cfg.lr))

        # positive and negative sample pair index matrix
        mask = torch.ones([node_num * 2, node_num * 2]) - torch.eye(node_num * 2)
        mask = mask.to(self.device)
        # training
        for epoch in range(0, cfg.max_epoch):
            # train mode
            self.train()

            # encoding with Eq. (3)-(5)
            Z1, Z2, E1, E2 = self.forward(data)

            # calculate comprehensive similarity by Eq. (6)
            S = comprehensive_similarity(Z1, Z2, E1, E2, self.alpha)

            # calculate hard sample aware contrastive loss by Eq. (10)-(11)
            loss = hard_sample_aware_infoNCE(S, mask, self.pos_neg_weight, self.pos_weight, node_num)

            # optimization
            loss.backward()
            optimizer.step()
            self.loss_curve.append(loss.item())
            self.logger.loss(epoch, loss)

            # testing and update weights of sample pairs
            if epoch % 10 == 0:
                self.eval()
                # encoding
                Z1, Z2, E1, E2 = self.forward(data)
                # calculate comprehensive similarity by Eq. (6)
                S = comprehensive_similarity(Z1, Z2, E1, E2, self.alpha)
                # fusion and testing
                embedding = (Z1 + Z2) / 2
                P, center = phi(embedding, self.cfg.dataset.n_clusters)

                # select high confidence samples
                H, H_mat = self.high_confidence(embedding, center)

                # calculate new weight of sample pair by Eq. (9)
                M, M_mat = self.pseudo_matrix(P, S, node_num)

                # update weight
                self.pos_weight[H] = M[H].data
                self.pos_neg_weight[H_mat] = M_mat[H_mat].data
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

    def get_embedding(self, data) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            self.eval()
            # encoding
            Z1, Z2, E1, E2 = self.forward(data)
            # calculate comprehensive similarity by Eq. (6)
            S = comprehensive_similarity(Z1, Z2, E1, E2, self.alpha)
            # fusion and testing
            embedding = (Z1 + Z2) / 2
        return embedding.detach(), S.detach()

    def clustering(self, data) -> Tuple[Tensor, Tensor, Tensor]:
        embedding, S = self.get_embedding(data)
        labels_, clustering_centers_ = KMeansGPU(self.cfg.dataset.n_clusters).fit(embedding)
        return embedding, labels_, clustering_centers_

    def evaluate(self, data):
        embedding, predicted_labels, clustering_centers = self.clustering(data)
        ground_truth = data.y.numpy()
        metric = DGCMetric(ground_truth, predicted_labels.numpy(), embedding, data.edge_index)
        results = metric.evaluate_one_epoch(self.logger, acc=True, nmi=True, ari=True, f1=True, hom=True, com=True, pur=True, sc=True, gre=True)
        return embedding, predicted_labels, results
