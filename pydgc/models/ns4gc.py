# -*- coding: utf-8 -*-
from typing import Tuple, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch import Tensor
from torch_geometric.data import Data

from ..clusterings import KMeansGPU
from ..metrics import DGCMetric
from ..utils import Logger
from yacs.config import CfgNode as CN
from . import DGCModel


def mask_feat(X: torch.Tensor, mask_prob: float):
    """Mask feature.

    Args:
        X (torch.Tensor): Feature matrix.
        mask_prob (float): Mask probability.

    Returns:
        torch.Tensor: Masked feature matrix.
    """
    drop_mask = (
            torch.empty((X.size(1),), dtype=torch.float32, device=X.device).uniform_()
            < mask_prob
    )
    X = X.clone()
    X[:, drop_mask] = 0

    return X


def drop_edge(A: torch.sparse.Tensor, drop_prob: float):
    """Drop edge with drop probability

    Args:
        A (torch.sparse.Tensor): Adjacency matrix.
        drop_prob (float): Drop probability.

    Returns:
        torch.sparse.Tensor: Dropped adjacency matrix.
    """
    n_edges = A._nnz()
    mask_rates = torch.full((n_edges,), fill_value=drop_prob,
                            dtype=torch.float)
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)

    E = A._indices()
    V = A._values()

    E = E[:, mask_idx]
    V = V[mask_idx]
    A = torch.sparse_coo_tensor(E, V, A.shape, device=A.device)

    return A


def add_self_loop(A: torch.sparse.Tensor):
    """Add self loop to the adjacency matrix.

    Args:
        A (torch.sparse.Tensor): Adjacency matrix.

    Returns:
        torch.sparse.Tensor: Adjacency matrix with self loop.
    """
    return A + sparse_identity(A.shape[0], device=A.device)


def normalize(A: torch.sparse.Tensor, add_self_loops=True, returnA=False):
    """Normalized the graph's adjacency matrix in the torch.sparse.Tensor format.

    Args:
        A (torch.sparse.Tensor): Adjacency matrix.
        add_self_loops (bool): Whether to add self loops.
        returnA (bool): Whether to return the original adjacency matrix.

    Returns:
        torch.sparse.Tensor: Normalized adjacency matrix.
    """
    if add_self_loops:
        A_hat = add_self_loop(A)
    else:
        A_hat = A

    D_hat_invsqrt = torch.sparse.sum(A_hat, dim=0).to_dense() ** -0.5
    D_hat_invsqrt[D_hat_invsqrt == torch.inf] = 0
    D_hat_invsqrt = sparse_diag(D_hat_invsqrt)
    A_norm = D_hat_invsqrt @ A_hat @ D_hat_invsqrt
    if returnA:
        return A_hat, A_norm
    else:
        return A_norm


def sparse_identity(dim, device):
    """Create a sparse identity matrix.

    Args:
        dim (int): Dimension of the identity matrix.
        device (torch.device): Device to create the matrix on.

    Returns:
        torch.sparse.Tensor: Sparse identity matrix.
    """
    indices = torch.arange(dim).unsqueeze(0).repeat(2, 1)
    values = torch.ones(dim)
    identity_matrix = torch.sparse_coo_tensor(indices, values,
                                              size=(dim, dim), device=device)
    return identity_matrix


def sparse_diag(V: torch.Tensor):
    """Create a sparse diagonal matrix.

    Args:
        V (torch.Tensor): Diagonal values.

    Returns:
        torch.sparse.Tensor: Sparse diagonal matrix.
    """
    size = V.size(0)
    indices = torch.arange(size).unsqueeze(0).repeat(2, 1)
    values = V
    diagonal_matrix = torch.sparse_coo_tensor(indices, values,
                                              size=(size, size), device=V.device)
    return diagonal_matrix


def augment(A: torch.sparse.Tensor, X: torch.Tensor,
            edge_mask_rate: float, feat_drop_rate: float):
    """Augment the graph and feature matrix.

    Args:
        A (torch.sparse.Tensor): Adjacency matrix.
        X (torch.Tensor): Feature matrix.
        edge_mask_rate (float): Edge mask rate.
        feat_drop_rate (float): Feature drop rate.

    Returns:
        torch.sparse.Tensor: Augmented adjacency matrix.
        torch.Tensor: Augmented feature matrix.
    """
    A = drop_edge(A, edge_mask_rate)
    X = mask_feat(X, feat_drop_rate)

    return A, X


class GCNConv(nn.Module):
    """Implementation of Graph Convolutional Network (GCN) layer.

    Args:
        in_dim (int): Input dimensionality of the layer.
        out_dim (int): Output dimensionality of the layer.
        activation (callable, optional): Activation function to use for the final representations. Defaults to None.
    """

    def __init__(self, in_dim, out_dim, activation=None):
        """Initializes the layer with specified parameters."""
        super(GCNConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = activation

        self.weight = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        self.bias = nn.Parameter(torch.FloatTensor(out_dim))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, A_norm, X):
        """Computes GCN representations according to input features and input graph.

        Args:
            A_norm (torch.sparse.Tensor): Normalized (n*n) sparse graph adjacency matrix.
            X (torch.Tensor): (n*in_dim) node feature matrix.

        Returns:
            torch.Tensor: An (n*out_dim) node representation matrix.
        """
        assert isinstance(X, torch.Tensor)
        assert isinstance(A_norm, torch.sparse.Tensor)

        output = torch.matmul(X, self.weight)
        output = torch.spmm(A_norm, output) + self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output


class NS4GC(DGCModel):
    """Reliable Node Similarity Matrix Guided Contrastive Graph Clustering.
    
    Reference: https://ieeexplore.ieee.org/abstract/document/10614738/

    Args:
        logger (Logger): Logger object.
        cfg (CN): Configuration object.
    """

    def __init__(self, logger: Logger, cfg: CN):
        super(NS4GC, self).__init__(logger, cfg)
        dims = cfg.model.dims.copy()
        dims.insert(0, cfg.dataset.num_features)
        self.encoder = nn.ModuleList()
        if len(dims) > 2:
            for i in range(len(dims) - 2):
                self.encoder.append(GCNConv(dims[i], dims[i + 1], activation=F.selu))
        self.encoder.append(GCNConv(dims[-2], dims[-1], activation=None))

        self.encoder = self.encoder.to(self.device)

        self.loss_curve = []
        self.nmi_curve = []
        self.pretrain_loss_curve = []
        self.best_embedding = None
        self.best_predicted_labels = None
        self.best_results = {'ACC': -1}

    def reset_parameters(self):
        pass

    def embed(self, A_norm, X):
        Z = X
        for layer in self.encoder:
            Z = layer(A_norm, Z)
        Z = F.normalize(Z, p=2, dim=1)
        return Z

    def forward(self, A_norm1, A_norm2, X1, X2) -> Any:
        Z1 = self.embed(A_norm1, X1)
        Z2 = self.embed(A_norm2, X2)
        return Z1, Z2

    def loss(self, *args, **kwargs) -> Tensor:
        pass

    def train_model(self, data: Data, cfg: CN = None, flag: str = "TRAIN NS4GC") -> Tuple[List, List, Tensor, Tensor, Dict]:
        if cfg is None:
            cfg = self.cfg.train
        self.logger.flag(flag)
        A = data.A.to(self.device)
        x = data.x.to(self.device)
        mask = data.mask.to(self.device)
        src, dst = data.edge_index[0], data.edge_index[1]
        pd1, pd2, pm1, pm2 = self.cfg.dataset.augmentation.pd1, self.cfg.dataset.augmentation.pd2, self.cfg.dataset.augmentation.pm1, self.cfg.dataset.augmentation.pm2
        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
        for epoch in range(1, cfg.max_epoch+1):
            optimizer.zero_grad()
            self.encoder.train()
            A1, X1 = augment(A, x, pd1, pm1)
            A2, X2 = augment(A, x, pd2, pm2)
            A_norm1 = normalize(A1, add_self_loops=True)
            A_norm2 = normalize(A2, add_self_loops=True)
            Z1, Z2 = self.forward(A_norm1, A_norm2, X1, X2)

            # Z1 and Z2 have been normalized.
            S = Z1 @ Z2.T

            loss_ali = - torch.diag(S).mean()

            loss_nei = - S[src, dst].mean()

            S = torch.masked_select(S, mask)
            S = torch.sigmoid((S - float(cfg.s)) / float(cfg.tau))
            loss_spa = S.mean()

            loss = loss_ali + float(cfg.lam) * loss_nei + float(cfg.gam) * loss_spa

            loss.backward()
            optimizer.step()
            self.logger.loss(epoch, loss)
            self.loss_curve.append(loss.item())
            if epoch % 1 == 0:
                if self.cfg.evaluate.each:
                    embedding, predicted_labels, results = self.evaluate(data)
                    self.nmi_curve.append(results['NMI'])
                    if results['ACC'] > self.best_results['ACC']:
                        self.best_embedding = embedding
                        self.best_predicted_labels = predicted_labels
                        self.best_results = results
        if not self.cfg.evaluate.each:
            embedding, predicted_labels, results = self.evaluate(data)
            return self.loss_curve, self.nmi_curve, embedding, predicted_labels, results
        return self.loss_curve, self.nmi_curve, self.best_embedding, self.best_predicted_labels, self.best_results

    def get_embedding(self, data: Data) -> Tensor:
        A = data.A.to(self.device)
        X = data.x.to(self.device)
        A_norm = normalize(A, add_self_loops=True)
        with torch.no_grad():
            self.eval()
            embedding = self.embed(A_norm, X)
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
        results = metric.evaluate_one_epoch(self.logger, self.cfg.evaluate)
        return embedding, predicted_labels, results
