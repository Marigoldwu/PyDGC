# -*- coding: utf-8 -*-
from typing import Tuple, Any
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from yacs.config import CfgNode as CN
from torch.nn import Module, Parameter
import scipy.sparse as sp
from torch import nn, Tensor
from torch.nn import Linear

from pydgc.metrics import DGCMetric
from pydgc.models import DGCModel
from pydgc.utils import Logger


def new_graph(edge_index, weight, n, device):
    edge_index = edge_index.cpu().numpy()
    indices = torch.from_numpy(
        np.vstack((edge_index[0], edge_index[1])).astype(np.int64)).to(device)
    values = weight
    shape = torch.Size((n, n))
    return torch.sparse_coo_tensor(indices, values, shape)


def normalize(mx):
    row_sum = np.array(mx.sum(1))
    r_inv = np.power(row_sum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


class GNNLayer(Module):

    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act = nn.Tanh()
        self.w = Parameter(torch.FloatTensor(out_features, in_features))
        torch.nn.init.xavier_uniform_(self.w)

    def forward(self, features, adj, active):
        if active:
            support = self.act(F.linear(features, self.w))  # add bias
        else:
            support = F.linear(features, self.w)  # add bias
        output = torch.mm(adj, support)
        return output


class IGAE_encoder(nn.Module):

    def __init__(self, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, n_input):
        super(IGAE_encoder, self).__init__()
        self.gnn_1 = GNNLayer(n_input, gae_n_enc_1)
        self.gnn_2 = GNNLayer(gae_n_enc_1, gae_n_enc_2)
        self.gnn_3 = GNNLayer(gae_n_enc_2, gae_n_enc_3)
        self.s = nn.Sigmoid()

    def forward(self, x, adj):
        z = self.gnn_1(x, adj, active=True)
        z = self.gnn_2(z, adj, active=True)
        z_igae = self.gnn_3(z, adj, active=False)

        return z_igae


class Cluster_layer(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(Cluster_layer, self).__init__()
        self.linear = nn.Sequential(nn.Linear(in_dims, out_dims),
                                    nn.Softmax(dim=1))

    def forward(self, h):
        c = self.linear(h)
        return c


class IGAE(nn.Module):
    def __init__(self, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, n_input, clusters):
        super(IGAE, self).__init__()
        self.encoder = IGAE_encoder(
            gae_n_enc_1=gae_n_enc_1,
            gae_n_enc_2=gae_n_enc_2,
            gae_n_enc_3=gae_n_enc_3,
            n_input=n_input,
        )
        self.cluster = Cluster_layer(
            in_dims=gae_n_enc_3,
            out_dims=clusters,
        )

    def forward(self, x, adj):

        z_igae = self.encoder(x, adj)

        c = self.cluster(z_igae)

        return z_igae, c

    @staticmethod
    def calc_loss(x, x_aug, temperature=0.2, sym=True):

        batch_size = x.shape[0]
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)

        sim_matrix = torch.exp(sim_matrix / temperature)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]

        if sym:

            loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
            loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
            #    print(pos_sim,sim_matrix.sum(dim=0))
            loss_0 = - torch.log(loss_0).mean()
            loss_1 = - torch.log(loss_1).mean()
            loss = (loss_0 + loss_1) / 2.0
        else:
            loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
            loss = - torch.log(loss).mean()

        return loss


class ViewLearner(nn.Module):
    def __init__(self, encoder, embedding_dim):
        super(ViewLearner, self).__init__()

        self.encoder = encoder
        self.input_dim = embedding_dim

        self.mlp_edge_model = torch.nn.Sequential(
            Linear(self.input_dim * 2, 1)
        )
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, adj, edge_index):
        node_emb = self.encoder(x, adj)
        src, dst = edge_index[0], edge_index[1]
        emb_src = node_emb[src]
        emb_dst = node_emb[dst]
        edge_emb = torch.cat([emb_src, emb_dst], 1)
        edge_logits = self.mlp_edge_model(edge_emb)

        return edge_logits


class AGCDRR(DGCModel):
    def __init__(self, logger: Logger, cfg: CN):
        super(AGCDRR, self).__init__(logger, cfg)
        igae_dims = cfg.model.dims.copy()
        igae_dims.insert(0, cfg.dataset.augmentation.pca_dim)
        self.igae = IGAE(
            gae_n_enc_1=igae_dims[1],
            gae_n_enc_2=igae_dims[2],
            gae_n_enc_3=igae_dims[3],
            n_input=igae_dims[0],
            clusters=cfg.dataset.n_clusters,
        ).to(self.device)

        self.view_learner = ViewLearner(
            IGAE_encoder(gae_n_enc_1=igae_dims[1],
                         gae_n_enc_2=igae_dims[2],
                         gae_n_enc_3=igae_dims[3],
                         n_input=igae_dims[0]),
            embedding_dim=igae_dims[-1]
        ).to(self.device)

        self.loss_curve = []
        self.best_embedding = None
        self.best_predicted_labels = None
        self.best_results = {'ACC': -1}

    def reset_parameters(self):
        pass

    def forward(self, *args, **kwargs) -> Any:
        pass

    def loss(self, *args, **kwargs) -> Tensor:
        pass

    def train_model(self, data: Data, cfg: CN = None, flag: str = "TRAIN AGCDRR"):
        if cfg is None:
            cfg = self.cfg.train
        view_optimizer = torch.optim.Adam(self.view_learner.parameters(), lr=float(cfg.view_lr))
        optimizer = torch.optim.Adam(self.igae.parameters(), lr=float(cfg.lr))
        adj = data.adj.to(self.device)
        x = data.x.to(self.device).float()

        for epoch in range(1, cfg.max_epoch + 1):
            self.view_learner.train()
            self.view_learner.zero_grad()
            self.igae.eval()
            z_igae, c = self.igae(x, adj)

            n = z_igae.shape[0]
            edge_logits = self.view_learner(x, adj, data.edge_index)

            batch_aug_edge_weight = torch.sigmoid(edge_logits).squeeze()  # p

            aug_adj = new_graph(data.edge_index.clone().detach().to(self.device), batch_aug_edge_weight, n, self.device)
            aug_adj = aug_adj.to_dense()
            aug_adj = aug_adj * adj
            aug_adj = aug_adj.cpu().detach().numpy() + np.eye(n)
            aug_adj = torch.from_numpy(normalize(aug_adj)).to(torch.float32).to(self.device)

            aug_z_igae, aug_c = self.igae(x, aug_adj)

            edge_drop_out_prob = 1 - batch_aug_edge_weight
            reg = edge_drop_out_prob.mean()

            view_loss = -1 * (
                        (cfg.reg_lambda * reg) + self.igae.calc_loss(c.T, aug_c.T) + self.igae.calc_loss(c, aug_c))

            view_loss.backward()
            view_optimizer.step()

            self.view_learner.eval()

            self.igae.train()
            self.igae.zero_grad()
            z_igae, c = self.igae(x, adj)

            n = z_igae.shape[0]
            edge_logits = self.view_learner(x, adj, data.edge_index)

            batch_aug_edge_weight = torch.sigmoid(edge_logits).squeeze()  # p

            aug_adj = new_graph(data.edge_index.clone().detach().to(self.device), batch_aug_edge_weight, n, self.device)
            aug_adj = aug_adj.to_dense()
            aug_adj = aug_adj * adj
            aug_adj = aug_adj.cpu().detach().numpy() + np.eye(n)
            aug_adj = torch.from_numpy(normalize(aug_adj)).to(torch.float32).to(self.device)

            aug_z_igae, aug_c = self.igae(x, aug_adj)

            z_mat = torch.matmul(z_igae, aug_z_igae.T)

            model_loss = self.igae.calc_loss(c.T, aug_c.T) + F.mse_loss(z_mat,
                                                                        torch.eye(n).to('cuda')) + self.igae.calc_loss(
                c, aug_c)
            model_loss.backward()
            optimizer.step()
            self.logger.loss(epoch, model_loss)
            self.loss_curve.append(model_loss.item())
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
        adj = data.adj.to(self.device)
        x = data.x.to(self.device).float()
        with torch.no_grad():
            self.view_learner.eval()
            self.igae.eval()
            z_igae, c = self.igae(x, adj)

            n = z_igae.shape[0]
            edge_logits = self.view_learner(x, adj, data.edge_index)

            batch_aug_edge_weight = torch.sigmoid(edge_logits).squeeze()  # p

            aug_adj = new_graph(data.edge_index.clone().detach().to(self.device), batch_aug_edge_weight, n, self.device)
            aug_adj = aug_adj.to_dense()
            aug_adj = aug_adj * adj
            aug_adj = aug_adj.cpu().detach().numpy() + np.eye(n)
            aug_adj = torch.from_numpy(normalize(aug_adj)).to(torch.float32).to(self.device)

            aug_z_igae, aug_c = self.igae(x, aug_adj)
            embedding = (c + aug_c) / 2
        return embedding.detach()

    def clustering(self, data) -> Tuple[Tensor, Tensor, None]:
        embedding = self.get_embedding(data)
        labels_ = embedding.argmax(dim=-1).detach().cpu()
        return embedding, labels_, None

    def evaluate(self, data):
        embedding, predicted_labels, _ = self.clustering(data)
        ground_truth = data.y.numpy()
        metric = DGCMetric(ground_truth, predicted_labels.numpy(), embedding, data.edge_index)
        results = metric.evaluate_one_epoch(self.logger, acc=True, nmi=True, ari=True, f1=True, hom=True, com=True,
                                            pur=True,
                                            sc=True, gre=True)
        return embedding, predicted_labels, results
