# -*- coding: utf-8 -*-
from typing import Tuple, List, Any

from torch import Tensor

from . import DGCModel
from yacs.config import CfgNode as CN

from ..utils import Logger
import torch.nn as nn
import torch
import numpy as np
from torch.nn import Embedding

try:
    import torch_cluster  # noqa

    random_walk = torch.ops.torch_cluster.random_walk
except ImportError:
    random_walk = None

from torch.utils.data import DataLoader
from torch_sparse import SparseTensor
from torch.nn.functional import normalize
import torch.nn.functional as F

from torch_geometric.utils.num_nodes import maybe_num_nodes


class Node2Vec(nn.Module):
    """
    Adapted from: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/node2vec.html

    The Node2Vec model from the
    `"node2vec: Scalable Feature Learning for Networks"
    <https://arxiv.org/abs/1607.00653>`_ paper where random walks of
    length :obj:`walk_length` are sampled in a given graph, and node embeddings
    are learned via negative sampling optimization.

    """

    def __init__(self, input_dim, hidden_dims, num_points, dropout, edge_index, embedding_dim, walk_length,
                 context_size,
                 walks_per_node=1, p=1, q=1, num_negative_samples=1,
                 num_nodes=None, sparse=False):
        super(Node2Vec, self).__init__()

        if random_walk is None:
            raise ImportError('`Node2Vec` requires `torch-cluster`.')

        N = maybe_num_nodes(edge_index, num_nodes)
        row, col = edge_index
        self.adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
        self.adj = self.adj.to('cpu')

        assert walk_length >= context_size

        self.embedding_dim = embedding_dim
        self.walk_length = walk_length - 1
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.p = p
        self.q = q
        self.num_negative_samples = num_negative_samples

        self.clus = None
        self.embedding = None
        # self.embedding = torch.empty(num_points, hidden_dims).to(device)

        self.w1 = nn.Linear(input_dim, hidden_dims, bias=True)
        self.w2 = nn.Linear(input_dim, hidden_dims, bias=True)
        self.w1.bias.data.fill_(0.0)
        self.w2.bias.data.fill_(0.0)

        self.iden = nn.Parameter(data=torch.randn((num_points, hidden_dims), dtype=torch.float).to(device),
                                 requires_grad=True)

    def get_embedding(self, X1, X2):
        return normalize(self.w1(X1) + self.w2(X2) + self.iden)

    def update_B(self, X1, X2, unique=None):
        self.embedding = normalize(self.w1(X1) + self.w2(X2) + F.embedding(unique, self.iden))

    def loader(self, **kwargs):
        return DataLoader(range(self.adj.sparse_size(0)),
                          collate_fn=self.sample, **kwargs)

    def pos_sample(self, batch):
        batch = batch.repeat(self.walks_per_node)
        rowptr, col, _ = self.adj.csr()
        rw = random_walk(rowptr, col, batch, self.walk_length, self.p, self.q)
        if not isinstance(rw, torch.Tensor):
            rw = rw[0]

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)

    def neg_sample(self, batch):
        batch = batch.repeat(self.walks_per_node * self.num_negative_samples)
        if self.clus == None:
            rw = torch.randint(self.adj.sparse_size(0), (batch.size(0), self.walk_length * self.num_negative_samples))
        else:
            rw = torch.empty(batch.size(0), self.walk_length * self.num_negative_samples, dtype=torch.long)
            for i in range(self.clus.max() + 1):
                cluster = (self.clus[batch.view(-1)] == i).nonzero(as_tuple=True)[0].tolist()
                neg_ind = (self.clus != i).nonzero(as_tuple=True)[0]
                rw[cluster] = neg_ind[
                    torch.randint(neg_ind.size(0), (len(cluster), self.walk_length * self.num_negative_samples))]
        rw = torch.cat([batch.view(-1, 1), rw], dim=-1)

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)

    def sample(self, batch):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)
        return self.pos_sample(batch), self.neg_sample(batch)

    def loss(self, pos_rw, neg_rw, mapping=None):
        r"""Computes the loss given positive and negative random walks."""

        pos_rw = F.embedding(pos_rw.view(-1), mapping.view(-1, 1)).view(pos_rw.size())
        neg_rw = F.embedding(neg_rw.view(-1), mapping.view(-1, 1)).view(neg_rw.size())
        start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()

        h_start = F.embedding(start, self.embedding).view(pos_rw.size(0), 1, self.embedding_dim)
        h_rest = F.embedding(rest.view(-1), self.embedding).view(pos_rw.size(0), -1, self.embedding_dim)

        out = (h_start * h_rest).sum(dim=-1)
        pos_loss = torch.logsumexp(out, dim=-1)

        start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()

        h_start = F.embedding(start, self.embedding).view(neg_rw.size(0), 1, self.embedding_dim)
        h_rest = F.embedding(rest.view(-1), self.embedding).view(neg_rw.size(0), -1, self.embedding_dim)
        out = (h_start * h_rest).sum(dim=-1)

        neg_loss = torch.logsumexp(out, dim=-1)
        neg_loss = torch.logsumexp(torch.cat((neg_loss.view(-1, 1), pos_loss.view(-1, 1)), dim=-1), dim=-1)
        # return -1*torch.mean(pos_loss - neg_loss)

        return -1 * torch.mean(torch.exp(pos_loss - neg_loss))


class S3GC(DGCModel):
    def __init__(self, logger: Logger, cfg: CN):
        super(S3GC, self).__init__(logger, cfg)
        self.model = Node2Vec(AX.size(-1), args.hidden_dims, AX.size(0), 0.0, edge_index, args.hidden_dims, args.walk_length,
                     args.walk_length, args.walks_per_node, p=1.0, q=1.0,
                     sparse=True).to(self.device)

    def reset_parameters(self):
        pass

    def forward(self, *args, **kwargs) -> Any:
        pass

    def loss(self, *args, **kwargs) -> Tensor:
        pass

    def train_model(self, *args, **kwargs) -> List:
        pass

    def get_embedding(self, *args, **kwargs) -> Tensor:
        pass

    def clustering(self, *args, **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
        pass

    def evaluate(self, *args, **kwargs):
        pass
