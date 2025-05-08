# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F

from .utils import *
from typing import List

layer_registry = LayerRegistry()


class BaseEncoder(nn.Module):
    def __init__(self,
                 dims: List[int] = None,
                 layer: str = 'linear',
                 act: str = 'relu',
                 act_last: bool = False,
                 add_self_loops: bool = True):
        """

        :param dims: A list of dimensions from input to output
        :param layer: type of layers, e.g., 'linear', 'gcn', 'gat', 'sage', 'sg'
        :param act: activation function, e.g., 'relu', ''
        :param act_last:
        :param add_self_loops:
        """
        super(BaseEncoder, self).__init__()
        self.act = act
        self.act_last = act_last
        self.add_self_loops = add_self_loops
        self.encoder = nn.Sequential()
        if not dims:
            raise ValueError("dims cannot be None and should be a list of dimensions from input to output")
        if len(dims) < 2:
            raise ValueError("dims must contain at least input and output dimensions")

        registered = layer_registry.list_layers()
        if layer not in registered:
            raise ValueError(f"Unsupported layer type: {layer}. Registered types: {registered}")
        LayerClass = layer_registry.get_layer(layer)
        self.LayerClass = LayerClass
        for i in range(len(dims) - 1):
            if layer == 'gcn':
                layer_instance = LayerClass(dims[i], dims[i + 1], add_self_loops=self.add_self_loops)
            else:
                layer_instance = LayerClass(dims[i], dims[i + 1])
            self.encoder.add_module(f'{layer}_{i}', layer_instance)
            if self.act_last or i < len(dims) - 2:
                self.encoder.add_module(f'{self.act}_{i}', self.act_func)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.encoder:
            if isinstance(layer, self.LayerClass):
                layer.reset_parameters()

    @property
    def act_func(self):
        if self.act == 'relu':
            return nn.ReLU()
        elif self.act == 'tanh':
            return nn.Tanh()
        elif self.act == 'sigmoid':
            return nn.Sigmoid()
        elif self.act == 'leaky_relu':
            return nn.LeakyReLU()
        elif self.act == 'elu':
            return nn.ELU()
        else:
            return nn.Identity()

    def forward(self, *args, **kwargs):
        raise NotImplementedError()


class MLPEncoder(BaseEncoder):
    def __init__(self, dims, act='relu', act_last=False):
        super(MLPEncoder, self).__init__(dims=dims,
                                         layer='linear',
                                         act=act,
                                         act_last=act_last)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

    def forward(self, x):
        encodes = []
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
            if isinstance(self.encoder[i], nn.Linear):
                encodes.append(x)
        return encodes


class GNNEncoder(BaseEncoder):
    def __init__(self, dims, layer='gcn', act='relu', act_last=False, add_self_loops=True):
        super(GNNEncoder, self).__init__(dims=dims,
                                         act=act,
                                         layer=layer,
                                         act_last=act_last,
                                         add_self_loops=add_self_loops)

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.encoder:
            if isinstance(layer, self.LayerClass):
                layer.reset_parameters()

    def forward(self, x, edge_index):
        for layer in self.encoder:
            if isinstance(layer, self.LayerClass):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        return x


class GATMEncoder(nn.Module):
    def __init__(self, dims, alpha=0.2):
        super(GATMEncoder, self).__init__()
        dims_next = dims.copy()
        dims_next = dims_next[1:] + [dims_next[0]]
        dims = list(zip(dims, dims_next))[:-1]
        self.encoder = nn.Sequential()
        for in_channel, out_channel in dims:
            self.encoder.append(GATMConv(in_channel, out_channel, alpha=alpha))
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.encoder:
            if isinstance(layer, GATMConv):
                layer.reset_parameters()

    def forward(self, x, adj, M):
        for layer in self.encoder:
            x = layer(x, adj, M)
        return x


class GATMConv(nn.Module):
    """
    Graph Attention Neural Network
    """

    def __init__(self, in_features, out_features, alpha=0.2):
        super(GATMConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        self.a_self = nn.Parameter(torch.zeros(size=(out_features, 1)))
        self.a_neighs = nn.Parameter(torch.zeros(size=(out_features, 1)))

        self.leaky_relu = nn.LeakyReLU(self.alpha)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_self.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_neighs.data, gain=1.414)

    def forward(self, x, adj, M, concat=True):
        h = torch.mm(x, self.W)
        attn_for_self = torch.mm(h, self.a_self)
        attn_for_neighs = torch.mm(h, self.a_neighs)
        attn_dense = attn_for_self + torch.transpose(attn_for_neighs, 0, 1)
        attn_dense = torch.mul(attn_dense, M)
        attn_dense = self.leaky_relu(attn_dense)

        zero_vec = -9e15 * torch.ones_like(adj)
        adj = torch.where(adj > 0, attn_dense, zero_vec)
        attention = F.softmax(adj, dim=1)
        h_prime = torch.matmul(attention, h)

        if concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
