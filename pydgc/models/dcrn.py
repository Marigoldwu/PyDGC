# -*- coding: utf-8 -*-
from typing import Tuple, List
import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch import Tensor
from torch.nn import Linear, Parameter, Module
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from . import DGCModel
from ..clusterings import KMeansGPU
from ..datasets import LoadAttribute
from ..metrics import DGCMetric
from ..utils import Logger, validate_and_create_path
from yacs.config import CfgNode as CN


def normalize_adj(adj, self_loop=True, symmetry=False):
    """
    normalize the adj matrix
    :param adj: input adj matrix
    :param self_loop: if add the self loop or not
    :param symmetry: symmetry normalize or not
    :return: the normalized adj matrix
    """
    # add the self_loop
    if self_loop:
        adj_tmp = adj + np.eye(adj.shape[0])
    else:
        adj_tmp = adj

    # calculate degree matrix and it's inverse matrix
    d = np.diag(adj_tmp.sum(0))
    d_inv = np.linalg.inv(d)

    # symmetry normalize: D^{-0.5} A D^{-0.5}
    if symmetry:
        sqrt_d_inv = np.sqrt(d_inv)
        norm_adj = np.matmul(np.matmul(sqrt_d_inv, adj_tmp), adj_tmp)

    # non-symmetry normalize: D^{-1} A
    else:
        norm_adj = np.matmul(d_inv, adj_tmp)

    return norm_adj


def numpy_to_torch(a, sparse=False):
    """
    numpy array to torch tensor
    :param a: the numpy array
    :param sparse: is sparse tensor or not
    :return: torch tensor
    """
    if sparse:
        a = torch.sparse.Tensor(a)
        a = a.to_sparse()
    else:
        a = torch.FloatTensor(a)
    return a


def remove_edge(A, similarity, remove_rate=0.1, device='cuda'):
    """
    remove edge based on embedding similarity
    Args:
        A: the origin adjacency matrix
        similarity: cosine similarity matrix of embedding
        remove_rate: the rate of removing linkage relation
        device: device
    Returns:
        Am: edge-masked adjacency matrix
    """
    # remove edges based on cosine similarity of embedding
    n_node = A.shape[0]
    for i in range(n_node):
        A[i, torch.argsort(similarity[i].cpu())[:int(round(remove_rate * n_node))]] = 0

    # normalize adj
    Am = normalize_adj(A, self_loop=True, symmetry=False)
    Am = numpy_to_torch(Am).to(device)
    return Am


# the reconstruction function from DFCN
def reconstruction_loss(X, A_norm, X_hat, Z_hat, A_hat):
    """
    reconstruction loss L_{}
    Args:
        X: the origin feature matrix
        A_norm: the normalized adj
        X_hat: the reconstructed X
        Z_hat: the reconstructed Z
        A_hat: the reconstructed A
    Returns: the reconstruction loss
    """
    loss_ae = F.mse_loss(X_hat, X)
    loss_w = F.mse_loss(Z_hat, torch.spmm(A_norm, X))
    loss_a = F.mse_loss(A_hat, A_norm.to_dense())
    loss_igae = loss_w + 0.1 * loss_a
    loss_rec = loss_ae + loss_igae
    return loss_rec


def target_distribution(Q):
    """
    calculate the target distribution (student-t distribution)
    Args:
        Q: the soft assignment distribution
    Returns: target distribution P
    """
    weight = Q ** 2 / Q.sum(0)
    P = (weight.t() / weight.sum(1)).t()
    return P


# clustering guidance from DFCN
def distribution_loss(Q, P):
    """
    calculate the clustering guidance loss L_{KL}
    Args:
        Q: the soft assignment distribution
        P: the target distribution
    Returns: L_{KL}
    """
    loss = F.kl_div((Q[0].log() + Q[1].log() + Q[2].log()) / 3, P, reduction='batchmean')
    return loss


def r_loss(AZ, Z):
    """
    the loss of propagated regularization (L_R)
    Args:
        AZ: the propagated embedding
        Z: embedding
    Returns: L_R
    """
    loss = 0
    for i in range(2):
        for j in range(3):
            p_output = F.softmax(AZ[i][j], dim=1)
            q_output = F.softmax(Z[i][j], dim=1)
            log_mean_output = ((p_output + q_output) / 2).log()
            loss += (F.kl_div(log_mean_output, p_output, reduction='batchmean') +
                     F.kl_div(log_mean_output, p_output, reduction='batchmean')) / 2
    return loss


def off_diagonal(x):
    """
    off-diagonal elements of x
    Args:
        x: the input matrix
    Returns: the off-diagonal elements of x
    """
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def cross_correlation(Z_v1, Z_v2):
    """
    calculate the cross-view correlation matrix S
    Args:
        Z_v1: the first view embedding
        Z_v2: the second view embedding
    Returns: S
    """
    return torch.mm(F.normalize(Z_v1, dim=1), F.normalize(Z_v2, dim=1).t())


def correlation_reduction_loss(S):
    """
    the correlation reduction loss L: MSE for S and I (identical matrix)
    Args:
        S: the cross-view correlation matrix S
    Returns: L
    """
    return torch.diagonal(S).add(-1).pow(2).mean() + off_diagonal(S).pow(2).mean()


def dicr_loss(name, Z_ae, Z_igae, AZ, Z, gamma_value):
    """
    Dual Information Correlation Reduction loss L_{DICR}
    Args:
        name: dataset name
        Z_ae: AE embedding including two-view node embedding [0, 1] and two-view cluster-level embedding [2, 3]
        Z_igae: IGAE embedding including two-view node embedding [0, 1] and two-view cluster-level embedding [2, 3]
        AZ: the propagated fusion embedding AZ
        Z: the fusion embedding Z
        gamma_value: gamma_value
    Returns:
        L_{DICR}
    """
    # Sample-level Correlation Reduction (SCR)
    # cross-view sample correlation matrix
    S_N_ae = cross_correlation(Z_ae[0], Z_ae[1])
    S_N_igae = cross_correlation(Z_igae[0], Z_igae[1])
    # loss of SCR
    L_N_ae = correlation_reduction_loss(S_N_ae)
    L_N_igae = correlation_reduction_loss(S_N_igae)

    # Feature-level Correlation Reduction (FCR)
    # cross-view feature correlation matrix
    S_F_ae = cross_correlation(Z_ae[2].t(), Z_ae[3].t())
    S_F_igae = cross_correlation(Z_igae[2].t(), Z_igae[3].t())

    # loss of FCR
    L_F_ae = correlation_reduction_loss(S_F_ae)
    L_F_igae = correlation_reduction_loss(S_F_igae)

    if name == "dblp" or name == "acm":
        L_N = 0.01 * L_N_ae + 10 * L_N_igae
        L_F = 0.5 * L_F_ae + 0.5 * L_F_igae
    else:
        L_N = 0.1 * L_N_ae + 5 * L_N_igae
        L_F = L_F_ae + L_F_igae

    # propagated regularization
    L_R = r_loss(AZ, Z)

    # loss of DICR
    loss_dicr = L_N + L_F + gamma_value * L_R

    return loss_dicr


def gaussian_noised_feature(X, device='cuda'):
    """
    add gaussian noise to the attribute matrix X
    Args:
        X: the attribute matrix
        device: device
    Returns: the noised attribute matrix X_tilde
    """
    N_1 = torch.Tensor(np.random.normal(1, 0.1, X.shape)).to(device)
    N_2 = torch.Tensor(np.random.normal(1, 0.1, X.shape)).to(device)
    X_tilde1 = X * N_1
    X_tilde2 = X * N_2
    return X_tilde1, X_tilde2


class AE_encoder(Module):
    def __init__(self, ae_n_enc_1, ae_n_enc_2, ae_n_enc_3, n_input, n_z):
        super(AE_encoder, self).__init__()
        self.enc_1 = Linear(n_input, ae_n_enc_1)
        self.enc_2 = Linear(ae_n_enc_1, ae_n_enc_2)
        self.enc_3 = Linear(ae_n_enc_2, ae_n_enc_3)
        self.z_layer = Linear(ae_n_enc_3, n_z)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        z = self.act(self.enc_1(x))
        z = self.act(self.enc_2(z))
        z = self.act(self.enc_3(z))
        z_ae = self.z_layer(z)
        return z_ae


class AE_decoder(Module):
    def __init__(self, ae_n_dec_1, ae_n_dec_2, ae_n_dec_3, n_input, n_z):
        super(AE_decoder, self).__init__()

        self.dec_1 = Linear(n_z, ae_n_dec_1)
        self.dec_2 = Linear(ae_n_dec_1, ae_n_dec_2)
        self.dec_3 = Linear(ae_n_dec_2, ae_n_dec_3)
        self.x_bar_layer = Linear(ae_n_dec_3, n_input)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, z_ae):
        z = self.act(self.dec_1(z_ae))
        z = self.act(self.dec_2(z))
        z = self.act(self.dec_3(z))
        x_hat = self.x_bar_layer(z)
        return x_hat


class AE(Module):
    def __init__(self, ae_n_enc_1, ae_n_enc_2, ae_n_enc_3, ae_n_dec_1, ae_n_dec_2, ae_n_dec_3, n_input, n_z, device='cuda'):
        super(AE, self).__init__()
        self.device = device

        self.encoder = AE_encoder(
            ae_n_enc_1=ae_n_enc_1,
            ae_n_enc_2=ae_n_enc_2,
            ae_n_enc_3=ae_n_enc_3,
            n_input=n_input,
            n_z=n_z).to(self.device)

        self.decoder = AE_decoder(
            ae_n_dec_1=ae_n_dec_1,
            ae_n_dec_2=ae_n_dec_2,
            ae_n_dec_3=ae_n_dec_3,
            n_input=n_input,
            n_z=n_z).to(self.device)
        self.loss_curve = []

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
                loss = 10 * F.mse_loss(x_hat, x)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
            self.loss_curve.append(loss_sum.item())
            logger.loss(epoch, loss_sum)

            with torch.no_grad():
                self.eval()
                _, embedding = self.forward(data.x)
                self.evaluate(logger, embedding, data.y, data.edge_index)

        pretrain_file_name = os.path.join(cfg.dir, "ae.pth")
        validate_and_create_path(pretrain_file_name)
        torch.save(self.state_dict(), pretrain_file_name)

    def evaluate(self, logger, embedding, y, edge_index):
        labels_, clustering_centers_ = KMeansGPU(3).fit(embedding)
        DGCMetric(y, labels_.cpu().numpy(), embedding, edge_index).evaluate_one_epoch(logger)


class GNNLayer(Module):
    def __init__(self, name, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.name = name
        self.in_features = in_features
        self.out_features = out_features
        if name == "dblp":
            self.act = nn.Tanh()
            self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        else:
            self.act = nn.Tanh()
            self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=False):
        if active:
            if self.name == "dblp":
                support = self.act(F.linear(features, self.weight))
            else:
                support = self.act(torch.mm(features, self.weight))
        else:
            if self.name == "dblp":
                support = F.linear(features, self.weight)
            else:
                support = torch.mm(features, self.weight)
        output = torch.spmm(adj, support)
        az = torch.spmm(adj, output)
        return output, az


class IGAE_encoder(Module):
    def __init__(self, name, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, n_input):
        super(IGAE_encoder, self).__init__()
        self.gnn_1 = GNNLayer(name, n_input, gae_n_enc_1)
        self.gnn_2 = GNNLayer(name, gae_n_enc_1, gae_n_enc_2)
        self.gnn_3 = GNNLayer(name, gae_n_enc_2, gae_n_enc_3)
        self.s = nn.Sigmoid()

    def forward(self, x, adj):
        z_1, az_1 = self.gnn_1(x, adj, active=True)
        z_2, az_2 = self.gnn_2(z_1, adj, active=True)
        z_igae, az_3 = self.gnn_3(z_2, adj, active=False)
        z_igae_adj = self.s(torch.mm(z_igae, z_igae.t()))
        return z_igae, z_igae_adj, [az_1, az_2, az_3], [z_1, z_2, z_igae]


class IGAE_decoder(nn.Module):
    def __init__(self, name, gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, n_input):
        super(IGAE_decoder, self).__init__()
        self.gnn_4 = GNNLayer(name, gae_n_dec_1, gae_n_dec_2)
        self.gnn_5 = GNNLayer(name, gae_n_dec_2, gae_n_dec_3)
        self.gnn_6 = GNNLayer(name, gae_n_dec_3, n_input)
        self.s = nn.Sigmoid()

    def forward(self, z_igae, adj):
        z_1, az_1 = self.gnn_4(z_igae, adj, active=True)
        z_2, az_2 = self.gnn_5(z_1, adj, active=True)
        z_hat, az_3 = self.gnn_6(z_2, adj, active=True)
        z_hat_adj = self.s(torch.mm(z_hat, z_hat.t()))
        return z_hat, z_hat_adj, [az_1, az_2, az_3], [z_1, z_2, z_hat]


class IGAE(nn.Module):
    def __init__(self, name, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, n_input, device='cuda'):
        super(IGAE, self).__init__()
        # IGAE encoder
        self.device = device
        self.encoder = IGAE_encoder(name,
                                    gae_n_enc_1=gae_n_enc_1,
                                    gae_n_enc_2=gae_n_enc_2,
                                    gae_n_enc_3=gae_n_enc_3,
                                    n_input=n_input).to(self.device)

        # IGAE decoder
        self.decoder = IGAE_decoder(name,
                                    gae_n_dec_1=gae_n_dec_1,
                                    gae_n_dec_2=gae_n_dec_2,
                                    gae_n_dec_3=gae_n_dec_3,
                                    n_input=n_input).to(self.device)

        self.loss_curve = []

    def forward(self, x, adj):
        x = x.to(self.device)
        adj = adj.to(self.device)
        z_igae, z_igae_adj, _, _ = self.encoder(x, adj)
        z_hat, z_hat_adj, _, _ = self.decoder(z_igae, adj)
        adj_hat = z_igae_adj + z_hat_adj
        return z_igae, z_hat, adj_hat

    def pretrain(self, logger: Logger, data: Data, cfg: CN = None, flag: str = "PRETRAIN IGAE"):
        if cfg is None:
            raise ValueError("Please provide a valid configuration for pretraining igae.")
        logger.flag(flag)
        data.x = data.x.to(self.device).float()
        adj = torch.from_numpy(normalize_adj(data.adj)).to(self.device).float()
        optimizer = torch.optim.Adam(self.parameters(), lr=float(cfg.lr))
        for epoch in range(1, cfg.max_epoch+1):
            self.train()
            optimizer.zero_grad()
            _, z_hat, adj_hat = self.forward(data.x, adj)
            loss_w = F.mse_loss(z_hat, torch.spmm(adj, data.x))
            loss_a = F.mse_loss(adj_hat, adj)

            loss = loss_w + float(cfg.gamma_value) * loss_a

            logger.loss(epoch, loss)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                self.eval()
                embedding, _, _ = self.forward(data.x, adj)
                self.evaluate(logger, embedding, data.y, data.edge_index)

        pretrain_file_name = os.path.join(cfg.dir, "igae.pth")
        torch.save(self.state_dict(), pretrain_file_name)

    def evaluate(self, logger, embedding, y, edge_index):
        labels_, clustering_centers_ = KMeansGPU(3).fit(embedding)
        DGCMetric(y, labels_.cpu().numpy(), embedding, edge_index).evaluate_one_epoch(logger)


class Readout(nn.Module):
    def __init__(self, K):
        super(Readout, self).__init__()
        self.K = K

    def forward(self, Z):
        # calculate cluster-level embedding
        Z_tilde = []

        # step1: split the nodes into K groups
        # step2: average the node embedding in each group
        n_node = Z.shape[0]
        step = n_node // self.K
        for i in range(0, n_node, step):
            if n_node - i < 2 * step:
                Z_tilde.append(torch.mean(Z[i:n_node], dim=0))
                break
            else:
                Z_tilde.append(torch.mean(Z[i:i + step], dim=0))

        # the cluster-level embedding
        Z_tilde = torch.cat(Z_tilde, dim=0)
        return Z_tilde.view(1, -1)


class DCRN(DGCModel):
    def __init__(self, logger: Logger, cfg: CN):
        super(DCRN, self).__init__(logger, cfg)
        # Auto Encoder
        ae_dims = cfg.model.dims.ae
        ae_dims.insert(0, self.cfg.dataset.augmentation.pca_dim)
        igae_dims = cfg.model.dims.igae
        igae_dims.insert(0, self.cfg.dataset.augmentation.pca_dim)
        name = cfg.dataset.name.lower().split("_")[0] if "_" in cfg.dataset.name else cfg.dataset.name.lower()
        self.name = name
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

        # fusion parameter from DFCN
        self.a = Parameter(nn.init.constant_(torch.zeros(n_node, n_z), 0.5), requires_grad=True).to(self.device)
        self.b = Parameter(nn.init.constant_(torch.zeros(n_node, n_z), 0.5), requires_grad=True).to(self.device)
        self.alpha = Parameter(torch.zeros(1)).to(self.device)
        # cluster layer (clustering assignment matrix)
        self.cluster_centers = Parameter(torch.Tensor(n_clusters, n_z), requires_grad=True).to(self.device)
        # readout function
        self.R = Readout(K=n_clusters)

        self.gamma = Parameter(torch.zeros(1)).to(self.device)
        self.loss_curve = []
        self.pretrain_loss_curve = []

    def reset_parameters(self):
        pass

    # calculate the soft assignment distribution Q
    def q_distribute(self, Z, Z_ae, Z_igae):
        """
        calculate the soft assignment distribution based on the embedding and the cluster centers
        Args:
            Z: fusion node embedding
            Z_ae: node embedding encoded by AE
            Z_igae: node embedding encoded by IGAE
        Returns:
            the soft assignment distribution Q
        """
        q = 1.0 / (1.0 + torch.sum(torch.pow(Z.unsqueeze(1) - self.cluster_centers, 2), 2))
        q = (q.t() / torch.sum(q, 1)).t()

        q_ae = 1.0 / (1.0 + torch.sum(torch.pow(Z_ae.unsqueeze(1) - self.cluster_centers, 2), 2))
        q_ae = (q_ae.t() / torch.sum(q_ae, 1)).t()

        q_igae = 1.0 / (1.0 + torch.sum(torch.pow(Z_igae.unsqueeze(1) - self.cluster_centers, 2), 2))
        q_igae = (q_igae.t() / torch.sum(q_igae, 1)).t()

        return [q, q_ae, q_igae]

    def forward(self, X_tilde1, Am, X_tilde2, Ad):
        # node embedding encoded by AE
        Z_ae1 = self.ae.encoder(X_tilde1)
        Z_ae2 = self.ae.encoder(X_tilde2)

        # node embedding encoded by IGAE
        Z_igae1, A_igae1, AZ_1, Z_1 = self.igae.encoder(X_tilde1, Am)
        Z_igae2, A_igae2, AZ_2, Z_2 = self.igae.encoder(X_tilde2, Ad)

        # cluster-level embedding calculated by readout function
        Z_tilde_ae1 = self.R(Z_ae1)
        Z_tilde_ae2 = self.R(Z_ae2)
        Z_tilde_igae1 = self.R(Z_igae1)
        Z_tilde_igae2 = self.R(Z_igae2)

        # linear combination of view 1 and view 2
        Z_ae = (Z_ae1 + Z_ae2) / 2
        Z_igae = (Z_igae1 + Z_igae2) / 2

        # node embedding fusion from DFCN
        Z_i = self.a * Z_ae + self.b * Z_igae
        Z_l = torch.spmm(Am, Z_i)
        S = torch.mm(Z_l, Z_l.t())
        S = F.softmax(S, dim=1)
        Z_g = torch.mm(S, Z_l)
        Z = self.alpha * Z_g + Z_l

        # AE decoding
        X_hat = self.ae.decoder(Z)

        # IGAE decoding
        Z_hat, Z_adj_hat, AZ_de, Z_de = self.igae.decoder(Z, Am)
        sim = (A_igae1 + A_igae2) / 2
        A_hat = sim + Z_adj_hat

        # node embedding and cluster-level embedding
        Z_ae_all = [Z_ae1, Z_ae2, Z_tilde_ae1, Z_tilde_ae2]
        Z_gae_all = [Z_igae1, Z_igae2, Z_tilde_igae1, Z_tilde_igae2]

        # the soft assignment distribution Q
        Q = self.q_distribute(Z, Z_ae, Z_igae)

        # propagated embedding AZ_all and embedding Z_all
        AZ_en = []
        Z_en = []
        for i in range(len(AZ_1)):
            AZ_en.append((AZ_1[i] + AZ_2[i]) / 2)
            Z_en.append((Z_1[i] + Z_2[i]) / 2)
        AZ_all = [AZ_en, AZ_de]
        Z_all = [Z_en, Z_de]

        return X_hat, Z_hat, A_hat, sim, Z_ae_all, Z_gae_all, Q, Z, AZ_all, Z_all

    def loss(self, *args, **kwargs) -> Tensor:
        pass

    def ae_igae_forward(self, x, adj):
        z_ae = self.ae.encoder(x)
        z_igae, z_igae_adj, _, _ = self.igae.encoder(x, adj)
        z_i = self.a * z_ae + self.b * z_igae
        z_l = torch.spmm(adj, z_i)
        s = torch.mm(z_l, z_l.t())
        s = F.softmax(s, dim=1)
        z_g = torch.mm(s, z_l)
        z_tilde = self.gamma * z_g + z_l
        x_hat = self.ae.decoder(z_tilde)
        z_hat, z_hat_adj, _, _ = self.igae.decoder(z_tilde, adj)
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
        data.x = data.x.to(self.device).float()
        adj = torch.from_numpy(normalize_adj(data.adj)).to(self.device).float()
        optimizer = torch.optim.Adam(params_to_optimize, lr=float(cfg.lr))
        for epoch in range(1, cfg.max_epoch + 1):
            self.ae.train()
            self.igae.train()
            optimizer.zero_grad()
            x_hat, z_hat, adj_hat, z_ae, z_igae, z_tilde = self.ae_igae_forward(data.x, adj)

            loss_1 = F.mse_loss(x_hat, data.x)
            loss_2 = F.mse_loss(z_hat, torch.spmm(adj, data.x))
            loss_3 = F.mse_loss(adj_hat, adj)

            loss_4 = F.mse_loss(z_ae, z_igae)
            loss = loss_1 + float(cfg.alpha) * loss_2 + float(cfg.beta) * loss_3 + float(cfg.omega) * loss_4

            self.logger.loss(epoch, loss)
            self.pretrain_loss_curve.append(loss.item())
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                x_hat, z_hat, adj_hat, z_ae, z_igae, embedding = self.ae_igae_forward(data.x, adj)
                labels_, clustering_centers_ = KMeansGPU(3).fit(embedding)
                DGCMetric(data.y, labels_.cpu().numpy(), embedding, data.edge_index).evaluate_one_epoch(self.logger)
        pretrain_ae_file_name = os.path.join(cfg.dir, f'ae_both.pth')
        pretrain_igae_file_name = os.path.join(cfg.dir, f'igae_both.pth')
        torch.save(self.ae.state_dict(), pretrain_ae_file_name)
        torch.save(self.igae.state_dict(), pretrain_igae_file_name)

    def train_model(self, data: Data, cfg: CN = None, flag: str = "TRAIN DCRN") -> List:
        if cfg is None:
            cfg = self.cfg.train
        # load pretrained ae model
        pretrain_ae_file_name = os.path.join(cfg.pretrain.both.dir, f'ae_both.pth')
        pretrain_igae_file_name = os.path.join(cfg.pretrain.both.dir, f'igae_both.pth')

        if not os.path.exists(pretrain_ae_file_name) or not os.path.exists(pretrain_igae_file_name):
            self.pretrain(data, cfg.pretrain.both, flag='PRETRAIN AE and IGAE')

        self.logger.flag(flag)
        self.ae.load_state_dict(torch.load(pretrain_ae_file_name, map_location=self.device))
        self.igae.load_state_dict(torch.load(pretrain_igae_file_name, map_location=self.device))

        X = data.x.to(self.device).float()
        A = data.adj
        Ad = torch.from_numpy(data.Ad).to(self.device).float()
        A_norm = torch.from_numpy(data.A_norm).to(self.device).float()
        with torch.no_grad():
            _, _, _, sim, _, _, _, embedding, _, _ = self.forward(X, A_norm, X, A_norm)
        labels_, clustering_centers_ = KMeansGPU(self.cfg.dataset.n_clusters).fit(embedding)
        DGCMetric(data.y.numpy(), labels_.detach().cpu().numpy(), embedding, data.edge_index).evaluate_one_epoch(self.logger)
        self.cluster_centers.data = clustering_centers_.to(self.device)

        # edge-masked adjacency matrix (Am): remove edges based on feature-similarity

        Am = remove_edge(A, sim, remove_rate=0.1)
        data.Am = Am
        # A = A.to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=float(cfg.lr))
        for epoch in range(1, cfg.max_epoch):
            # add gaussian noise to X
            self.train()
            optimizer.zero_grad()
            X_tilde1, X_tilde2 = gaussian_noised_feature(X, self.device)

            # input & output
            X_hat, Z_hat, A_hat, _, Z_ae_all, Z_gae_all, Q, Z, AZ_all, Z_all = self.forward(X_tilde1, Ad, X_tilde2, Am)

            # calculate loss: L_{DICR}, L_{REC} and L_{KL}
            L_DICR = dicr_loss(self.name, Z_ae_all, Z_gae_all, AZ_all, Z_all, float(cfg.gamma_value))
            L_REC = reconstruction_loss(X, A_norm, X_hat, Z_hat, A_hat)
            L_KL = distribution_loss(Q, target_distribution(Q[0].data))
            loss = L_DICR + L_REC + float(cfg.lambda_value) * L_KL

            loss.backward(retain_graph=True)
            optimizer.step()
            self.logger.loss(epoch, loss)
            self.loss_curve.append(loss.item())
            if self.cfg.evaluate.each:
                self.evaluate(data)
        return self.loss_curve

    def get_embedding(self, data) -> Tensor:
        Ad = torch.from_numpy(data.Ad).to(self.device).float()
        Am = data.Am.to(self.device)
        X = data.x.to(self.device).float()
        with torch.no_grad():
            X_tilde1, X_tilde2 = gaussian_noised_feature(X, self.device)
            _, _, _, _, _, _, _, embedding, _, _ = self.forward(X_tilde1, Ad, X_tilde2, Am)
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
