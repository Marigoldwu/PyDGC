# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import target_distribution
from torch.nn.parameter import Parameter


class SSCLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, method: str = 'kl_div', v: float = 1.0):
        """
        Self-supervised clustering layer

        :param in_channels: dimension of embeddings
        :param out_channels: number of clusters
        :param method: 'kl_div' for default, 'cross_entropy', 'mse'
        :param v: v=1.0 for default
        """
        super(SSCLayer, self).__init__()
        self.cluster_centers = Parameter(torch.Tensor(out_channels, in_channels))
        self.method = method
        self.v = v
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.cluster_centers.data)

    def forward(self, embedding):
        q = 1.0 / (1.0 + torch.sum(torch.pow(embedding.unsqueeze(1) - self.cluster_centers, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q

    def get_q(self, embedding):
        with torch.no_grad():
            self.eval()
            q = 1.0 / (1.0 + torch.sum(torch.pow(embedding.unsqueeze(1) - self.cluster_centers, 2), 2) / self.v)
            q = q.pow((self.v + 1.0) / 2.0)
            q = (q.t() / torch.sum(q, 1)).t()
        return q

    @staticmethod
    def loss(q, method='kl_div'):
        """
        Calculate loss of self-supervised clustering
        :param q:
        :param method:
        :return:
        """
        p = target_distribution(q.detach().data)
        if method == 'mse':
            return F.mse_loss(q, p)
        if method == 'cross_entropy':
            return F.cross_entropy(q.log(), p)
        return F.kl_div(q.log(), p, reduction='batchmean')
