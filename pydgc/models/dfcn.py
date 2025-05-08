# -*- coding: utf-8 -*-
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

from torch.nn import Linear
from typing import Tuple, List, Any

from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj

from ..clusterings import KMeansGPU
from ..datasets import LoadAttribute
from ..metrics import DGCMetric
from ..utils import Logger, validate_and_create_path
from yacs.config import CfgNode as CN
from torch import Tensor

from . import DGCModel


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


class AE_encoder(nn.Module):

    def __init__(self, ae_n_enc_1, ae_n_enc_2, ae_n_enc_3, n_input, n_z, device='cuda'):
        super(AE_encoder, self).__init__()
        self.device = device
        self.enc_1 = Linear(n_input, ae_n_enc_1).to(self.device)
        self.enc_2 = Linear(ae_n_enc_1, ae_n_enc_2).to(self.device)
        self.enc_3 = Linear(ae_n_enc_2, ae_n_enc_3).to(self.device)
        self.z_layer = Linear(ae_n_enc_3, n_z).to(self.device)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.enc_1.reset_parameters()
        self.enc_2.reset_parameters()
        self.enc_3.reset_parameters()
        self.z_layer.reset_parameters()

    def forward(self, x):
        x = x.to(self.device)
        z = self.act(self.enc_1(x))
        z = self.act(self.enc_2(z))
        z = self.act(self.enc_3(z))
        z_ae = self.z_layer(z)
        return z_ae


class AE_decoder(nn.Module):

    def __init__(self, ae_n_dec_1, ae_n_dec_2, ae_n_dec_3, n_input, n_z, device='cuda'):
        super(AE_decoder, self).__init__()
        self.device = device
        self.dec_1 = Linear(n_z, ae_n_dec_1).to(self.device)
        self.dec_2 = Linear(ae_n_dec_1, ae_n_dec_2).to(self.device)
        self.dec_3 = Linear(ae_n_dec_2, ae_n_dec_3).to(self.device)
        self.x_bar_layer = Linear(ae_n_dec_3, n_input).to(self.device)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.dec_1.reset_parameters()
        self.dec_2.reset_parameters()
        self.dec_3.reset_parameters()
        self.x_bar_layer.reset_parameters()

    def forward(self, z_ae):
        z_ae = z_ae.to(self.device)
        z = self.act(self.dec_1(z_ae))
        z = self.act(self.dec_2(z))
        z = self.act(self.dec_3(z))
        x_hat = self.x_bar_layer(z)
        return x_hat


class AE(nn.Module):

    def __init__(self, ae_n_enc_1, ae_n_enc_2, ae_n_enc_3, ae_n_dec_1, ae_n_dec_2, ae_n_dec_3, n_input, n_z, device='cuda'):
        super(AE, self).__init__()
        self.device = device
        self.encoder = AE_encoder(
            ae_n_enc_1=ae_n_enc_1,
            ae_n_enc_2=ae_n_enc_2,
            ae_n_enc_3=ae_n_enc_3,
            n_input=n_input,
            n_z=n_z).to(device)

        self.decoder = AE_decoder(
            ae_n_dec_1=ae_n_dec_1,
            ae_n_dec_2=ae_n_dec_2,
            ae_n_dec_3=ae_n_dec_3,
            n_input=n_input,
            n_z=n_z).to(device)

        self.loss_curve = []
        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()

    def forward(self, x):
        x = x.to(self.device)
        z_ae = self.encoder(x)
        x_hat = self.decoder(z_ae)
        return x_hat, z_ae

    def pretrain(self, logger, data: Data, cfg: CN = None, flag: str = "PRETRAIN AE"):
        if cfg is None:
            raise ValueError("Please provide a valid configuration for pretraining ae.")
        # when ae is trained in pre-training mode, cfg.pretrain must be input as parameter
        attribute = LoadAttribute(data.x)
        train_loader = DataLoader(attribute, batch_size=256, shuffle=True)
        logger.flag(flag)
        optimizer = torch.optim.Adam(self.parameters(), lr=float(cfg.lr))
        for epoch in range(1, cfg.max_epoch + 1):
            self.train()
            loss_sum = torch.tensor(0.0)
            for batch_idx, (x, _) in enumerate(train_loader):
                optimizer.zero_grad()
                x = x.to(self.device)
                x_hat, z_ae = self.forward(x)
                loss = F.mse_loss(x_hat, x)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
            self.loss_curve.append(loss_sum.item())
            logger.loss(epoch, loss_sum)

            # with torch.no_grad():
            #     self.eval()
            #     _, embedding = self.forward(data.x)
            #     self.evaluate(logger, embedding, data.y, data.edge_index)

        pretrain_file_name = os.path.join(cfg.dir, "ae.pth")
        validate_and_create_path(pretrain_file_name)
        torch.save(self.state_dict(), pretrain_file_name)

    # def evaluate(self, logger, embedding, y, edge_index):
    #     labels_, clustering_centers_ = KMeansGPU(3).fit(embedding)
    #     DGCMetric(y, labels_.cpu().numpy(), embedding, edge_index).evaluate_one_epoch(logger)


class GNNLayer(nn.Module):

    def __init__(self, name, in_features, out_features, device='cuda'):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.name = name
        self.device = device
        if name == "dblp" or name == "hhar":
            self.act = nn.Tanh()
            self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        elif name == "reut":
            self.act = nn.LeakyReLU(0.2, inplace=True)
            self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        else:
            self.act = nn.Tanh()
            self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=False):
        if active:
            if self.name == "dblp" or self.name == "hhar":
                support = self.act(F.linear(features, self.weight))
            else:
                support = self.act(torch.mm(features, self.weight))
        else:
            if self.name == "dblp" or self.name == "hhar":
                support = F.linear(features, self.weight)
            else:
                support = torch.mm(features, self.weight)
        output = torch.mm(adj, support)
        return output


class IGAE_encoder(nn.Module):

    def __init__(self, name, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, n_input, device='cuda'):
        super(IGAE_encoder, self).__init__()
        self.name = name
        self.device = device
        self.gnn_1 = GNNLayer(name, n_input, gae_n_enc_1).to(self.device)
        self.gnn_2 = GNNLayer(name, gae_n_enc_1, gae_n_enc_2).to(self.device)
        self.gnn_3 = GNNLayer(name, gae_n_enc_2, gae_n_enc_3).to(self.device)
        self.s = nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self):
        self.gnn_1.reset_parameters()
        self.gnn_2.reset_parameters()
        self.gnn_3.reset_parameters()

    def forward(self, x, adj):
        x, adj = x.to(self.device), adj.to(self.device)
        z = self.gnn_1(x, adj, active=False if self.name == "hhar" else True)
        z = self.gnn_2(z, adj, active=False if self.name == "hhar" else True)
        z_igae = self.gnn_3(z, adj, active=False)
        z_igae_adj = self.s(torch.mm(z_igae, z_igae.t()))
        return z_igae, z_igae_adj


class IGAE_decoder(nn.Module):

    def __init__(self, name, gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, n_input, device='cuda'):
        super(IGAE_decoder, self).__init__()
        self.name = name
        self.device = device
        self.gnn_4 = GNNLayer(name, gae_n_dec_1, gae_n_dec_2).to(self.device)
        self.gnn_5 = GNNLayer(name, gae_n_dec_2, gae_n_dec_3).to(self.device)
        self.gnn_6 = GNNLayer(name, gae_n_dec_3, n_input).to(self.device)
        self.s = nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self):
        self.gnn_4.reset_parameters()
        self.gnn_5.reset_parameters()
        self.gnn_6.reset_parameters()

    def forward(self, z_igae, adj):
        z_igae = z_igae.to(self.device)
        adj = adj.to(self.device)
        z = self.gnn_4(z_igae, adj, active=False if self.name == "hhar" else True)
        z = self.gnn_5(z, adj, active=False if self.name == "hhar" else True)
        z_hat = self.gnn_6(z, adj, active=False if self.name == "hhar" else True)
        z_hat_adj = self.s(torch.mm(z_hat, z_hat.t()))
        return z_hat, z_hat_adj


class IGAE(nn.Module):

    def __init__(self, name, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, n_input, device='cuda'):
        super(IGAE, self).__init__()
        self.device = device
        self.encoder = IGAE_encoder(name=name,
                                    gae_n_enc_1=gae_n_enc_1,
                                    gae_n_enc_2=gae_n_enc_2,
                                    gae_n_enc_3=gae_n_enc_3,
                                    n_input=n_input).to(device)

        self.decoder = IGAE_decoder(name=name,
                                    gae_n_dec_1=gae_n_dec_1,
                                    gae_n_dec_2=gae_n_dec_2,
                                    gae_n_dec_3=gae_n_dec_3,
                                    n_input=n_input).to(device)
        self.loss_curve = []
        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()

    def forward(self, x, adj):
        x = x.to(self.device)
        adj = adj.to(self.device)
        z_igae, z_igae_adj = self.encoder(x, adj)
        z_hat, z_hat_adj = self.decoder(z_igae, adj)
        adj_hat = z_igae_adj + z_hat_adj
        return z_igae, z_hat, adj_hat

    def pretrain(self, logger: Logger, data: Data, cfg: CN = None, flag: str = "PRETRAIN IGAE"):
        if cfg is None:
            raise ValueError("Please provide a valid configuration for pretraining igae.")
        logger.flag(flag)
        data.x = data.x.to(self.device)
        data.adj = data.adj.to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=float(cfg.lr))
        for epoch in range(1, cfg.max_epoch+1):
            self.train()
            optimizer.zero_grad()
            z_igae, z_hat, adj_hat = self.forward(data.x, data.adj)

            loss_w = F.mse_loss(z_hat, torch.spmm(data.adj, data.x))
            loss_a = F.mse_loss(adj_hat, data.adj.to_dense())
            loss = loss_w + float(cfg.gamma_value) * loss_a

            logger.loss(epoch, loss)
            loss.backward()
            optimizer.step()
            # with torch.no_grad():
            #     self.eval()
            #     embedding, _, _ = self.forward(data.x, data.adj)
            #     self.evaluate(logger, embedding, data.y, data.edge_index)

        pretrain_file_name = os.path.join(cfg.dir, "igae.pth")
        torch.save(self.state_dict(), pretrain_file_name)

    # def evaluate(self, logger, embedding, y, edge_index):
    #     labels_, clustering_centers_ = KMeansGPU(3).fit(embedding)
    #     DGCMetric(y, labels_.cpu().numpy(), embedding, edge_index).evaluate_one_epoch(logger)


class DFCN(DGCModel):
    def __init__(self, logger: Logger, cfg: CN):
        super(DFCN, self).__init__(logger, cfg)
        ae_dims = cfg.model.dims.ae
        ae_dims.insert(0, self.cfg.dataset.augmentation.pca_dim)
        igae_dims = cfg.model.dims.igae
        igae_dims.insert(0, self.cfg.dataset.augmentation.pca_dim)
        name = cfg.dataset.name.lower().split("_")[0] if "_" in cfg.dataset.name else cfg.dataset.name.lower()
        n_node = cfg.dataset.num_nodes
        n_z = ae_dims[-1]
        n_clusters = cfg.dataset.n_clusters

        self.ae = AE(ae_dims[1], ae_dims[2], ae_dims[3], ae_dims[3], ae_dims[2], ae_dims[1], ae_dims[0],
                     ae_dims[-1]).to(self.device)

        self.igae = IGAE(name=name,
                         gae_n_enc_1=igae_dims[1],
                         gae_n_enc_2=igae_dims[2],
                         gae_n_enc_3=igae_dims[3],
                         gae_n_dec_1=igae_dims[3],
                         gae_n_dec_2=igae_dims[2],
                         gae_n_dec_3=igae_dims[1],
                         n_input=igae_dims[0]).to(self.device)

        self.a = nn.Parameter(nn.init.constant_(torch.zeros(n_node, n_z), 0.5), requires_grad=True).to(self.device)
        self.b = 1 - self.a

        self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, n_z), requires_grad=True).to(self.device)
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        self.gamma = nn.Parameter(torch.zeros(1)).to(self.device)
        self.v = cfg.model.v
        self.loss_curve = []
        self.pretrain_loss_curve = []
        self.reset_parameters()

    def reset_parameters(self):
        self.ae.reset_parameters()
        self.igae.reset_parameters()
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def forward(self, data: Data) -> Any:
        x = data.x.to(self.device)
        adj = data.adj.to(self.device)

        z_ae = self.ae.encoder(x)
        z_igae, z_igae_adj = self.igae.encoder(x, adj)
        z_i = self.a * z_ae + self.b * z_igae
        z_l = torch.mm(adj, z_i)
        s = torch.mm(z_l, z_l.t())
        s = F.softmax(s, dim=1)
        z_g = torch.mm(s, z_l)
        z_tilde = self.gamma * z_g + z_l
        x_hat = self.ae.decoder(z_tilde)
        z_hat, z_hat_adj = self.igae.decoder(z_tilde, adj)
        adj_hat = z_igae_adj + z_hat_adj

        q = 1.0 / (1.0 + torch.sum(torch.pow(z_tilde.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        q1 = 1.0 / (1.0 + torch.sum(torch.pow(z_ae.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q1 = q1.pow((self.v + 1.0) / 2.0)
        q1 = (q1.t() / torch.sum(q1, 1)).t()

        q2 = 1.0 / (1.0 + torch.sum(torch.pow(z_igae.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q2 = q2.pow((self.v + 1.0) / 2.0)
        q2 = (q2.t() / torch.sum(q2, 1)).t()

        return x_hat, z_hat, adj_hat, z_ae, z_igae, q, q1, q2, z_tilde

    def loss(self, *args, **kwargs) -> Tensor:
        pass

    def ae_igae_forward(self, data):
        x = data.x.to(self.device)
        adj = data.adj.to(self.device)
        z_ae = self.ae.encoder(x)
        z_igae, z_igae_adj = self.igae.encoder(x, adj)
        z_i = self.a * z_ae + self.b * z_igae
        z_l = torch.spmm(adj, z_i)
        s = torch.mm(z_l, z_l.t())
        s = F.softmax(s, dim=1)
        z_g = torch.mm(s, z_l)
        z_tilde = self.gamma * z_g + z_l
        x_hat = self.ae.decoder(z_tilde)
        z_hat, z_hat_adj = self.igae.decoder(z_tilde, adj)
        adj_hat = z_igae_adj + z_hat_adj

        return x_hat, z_hat, adj_hat, z_ae, z_igae, z_tilde

    def pretrain(self, data: Data, cfg: CN = None, flag: str = "PRETRAIN AE_IGAE"):
        # pretrain.both: alpha, beta, omega, lr, max_epoch, dir
        if cfg is None:
            cfg = self.cfg.train.pretrain.both
        pretrain_ae_file_name = os.path.join(self.cfg.train.pretrain.ae.dir, f'ae.pth')
        pretrain_igae_file_name = os.path.join(self.cfg.train.pretrain.igae.dir, f'igae.pth')
        if not os.path.exists(pretrain_ae_file_name):
            self.ae.pretrain(self.logger, data, self.cfg.train.pretrain.ae, flag=f'PRETRAIN AE for AE_IGAE')
        if not os.path.exists(pretrain_igae_file_name):
            self.igae.pretrain(self.logger, data, self.cfg.train.pretrain.igae, flag=f'PRETRAIN IGAE for AE_IGAE')

        self.logger.flag(flag)
        self.reset_parameters()
        self.ae.load_state_dict(torch.load(pretrain_ae_file_name, map_location=self.device))
        self.igae.load_state_dict(torch.load(pretrain_igae_file_name, map_location=self.device))
        params_to_optimize = []
        params_to_optimize.extend(self.ae.parameters())
        params_to_optimize.extend(self.igae.parameters())
        data.x = data.x.to(self.device)
        data.adj = data.adj.to(self.device)
        optimizer = torch.optim.Adam(params_to_optimize, lr=float(cfg.lr))
        for epoch in range(1, cfg.max_epoch + 1):
            self.ae.train()
            self.igae.train()
            optimizer.zero_grad()
            x_hat, z_hat, adj_hat, z_ae, z_igae, z_tilde = self.ae_igae_forward(data)
            loss_1 = F.mse_loss(x_hat, data.x)
            loss_2 = F.mse_loss(z_hat, torch.spmm(data.adj, data.x))
            loss_3 = F.mse_loss(adj_hat, data.adj.to_dense())
            loss_4 = F.mse_loss(z_ae, z_igae)
            loss = loss_1 + float(cfg.alpha) * loss_2 + float(cfg.beta) * loss_3 + float(cfg.omega) * loss_4
            self.logger.loss(epoch, loss)
            self.pretrain_loss_curve.append(loss.item())
            loss.backward()
            optimizer.step()
            # with torch.no_grad():
            #     x_hat, z_hat, adj_hat, z_ae, z_igae, embedding = self.ae_igae_forward(data)
            #     labels_, clustering_centers_ = KMeansGPU(3).fit(embedding)
            #     DGCMetric(data.y, labels_.cpu().numpy(), embedding, data.edge_index).evaluate_one_epoch(self.logger)
        pretrain_ae_file_name = os.path.join(cfg.dir, f'ae_both.pth')
        pretrain_igae_file_name = os.path.join(cfg.dir, f'igae_both.pth')
        torch.save(self.ae.state_dict(), pretrain_ae_file_name)
        torch.save(self.igae.state_dict(), pretrain_igae_file_name)

    def train_model(self, data: Data, cfg: CN = None, flag: str = "TRAIN DFCN") -> List:
        if cfg is None:
            cfg = self.cfg.train
        # load pretrained ae model
        pretrain_ae_file_name = os.path.join(cfg.pretrain.both.dir, f'ae_both.pth')
        pretrain_igae_file_name = os.path.join(cfg.pretrain.both.dir, f'igae_both.pth')

        if not os.path.exists(pretrain_ae_file_name) or not os.path.exists(pretrain_igae_file_name):
            self.pretrain(data, cfg.pretrain.both, flag='PRETRAIN AE and IGAE')

        self.logger.flag(flag)
        self.reset_parameters()
        self.ae.load_state_dict(torch.load(pretrain_ae_file_name, map_location=self.device))
        self.igae.load_state_dict(torch.load(pretrain_igae_file_name, map_location=self.device))

        data.x = data.x.to(self.device)
        data.adj = data.adj.to(self.device)
        embedding = self.get_embedding(data)
        labels_, clustering_centers_ = KMeansGPU(self.cfg.dataset.n_clusters).fit(embedding)
        DGCMetric(data.y.numpy(), labels_.detach().cpu().numpy(), embedding, data.edge_index).evaluate_one_epoch(self.logger)
        self.cluster_layer.data = clustering_centers_.to(self.device)

        optimizer = torch.optim.Adam(self.parameters(), lr=float(cfg.lr))
        for epoch in range(1, cfg.max_epoch + 1):
            self.train()
            optimizer.zero_grad()
            x_hat, z_hat, adj_hat, z_ae, z_igae, q, q1, q2, z_tilde = self.forward(data)

            tmp_q = q.data
            p = target_distribution(tmp_q)

            loss_ae = F.mse_loss(x_hat, data.x)
            loss_w = F.mse_loss(z_hat, torch.spmm(data.adj, data.x))
            loss_a = F.mse_loss(adj_hat, data.adj.to_dense())
            loss_igae = loss_w + float(cfg.gamma_value) * loss_a
            loss_kl = F.kl_div((q.log() + q1.log() + q2.log()) / 3, p, reduction='batchmean')
            loss = loss_ae + loss_igae + float(cfg.lambda_value) * loss_kl
            loss.backward()
            optimizer.step()
            self.logger.loss(epoch, loss)
            self.loss_curve.append(loss.item())
            if self.cfg.evaluate.each:
                self.evaluate(data)
        return self.loss_curve

    def get_embedding(self, data) -> Tensor:
        with torch.no_grad():
            self.eval()
            x_hat, z_hat, adj_hat, z_ae, z_igae, q, q1, q2, embedding = self.forward(data)
            return embedding.detach()

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
        embedding, labels, clustering_centers = self.clustering(data)
        ground_truth = data.y.numpy()
        metric = DGCMetric(ground_truth, labels.numpy(), embedding, data.edge_index)
        metric.evaluate_one_epoch(self.logger, acc=True, nmi=True, ari=True, f1=True, hom=True, com=True, pur=True, sc=True, gre=True)
