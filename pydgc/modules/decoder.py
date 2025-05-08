# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from typing import List

from torch import Tensor

from . import layer_registry


class BaseDecoder(nn.Module):
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
        super(BaseDecoder, self).__init__()
        self.act = act
        self.act_last = act_last
        self.add_self_loops = add_self_loops
        self.decoder = nn.Sequential()
        if not dims:
            raise ValueError("dims cannot be None and should be a list of dimensions from input to output")
        if len(dims) < 2:
            raise ValueError("dims must contain at least input and output dimensions")

        registered = layer_registry.list_layers()
        if layer not in registered:
            raise ValueError(f"Unsupported layer type: {layer}. Registered types: {registered}")
        LayerClass = layer_registry.get_layer(layer)
        self.LayerClass = LayerClass
        # reverse dims
        dims = dims[::-1]
        for i in range(len(dims) - 1):
            if layer == 'gcn':
                layer_instance = LayerClass(dims[i], dims[i + 1], add_self_loops=self.add_self_loops)
            else:
                layer_instance = LayerClass(dims[i], dims[i + 1])
            self.decoder.add_module(f'{layer}_{i}', layer_instance)
            if self.act_last or i < len(dims) - 2:
                self.decoder.add_module(f'{self.act}_{i}', self.act_func)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.decoder:
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


class MLPDecoder(BaseDecoder):
    def __init__(self, dims, act='relu', act_last=False):
        super(MLPDecoder, self).__init__(dims=dims,
                                         layer='linear',
                                         act=act,
                                         act_last=act_last)

    def forward(self, x) -> List[Tensor]:
        decodes = []
        for i in range(len(self.decoder)):
            x = self.decoder[i](x)
            if isinstance(self.decoder[i], nn.Linear):
                decodes.append(x)
        return decodes


class GNNAttributeDecoder(BaseDecoder):
    def __init__(self, dims, layer='gcn', act='relu', act_last=False, add_self_loops=True):
        super(GNNAttributeDecoder, self).__init__(dims=dims,
                                                  act=act,
                                                  layer=layer,
                                                  act_last=act_last,
                                                  add_self_loops=add_self_loops)

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.decoder:
            if isinstance(layer, self.LayerClass):
                layer.reset_parameters()

    def forward(self, x, edge_index) -> Tensor:
        for layer in self.decoder:
            if isinstance(layer, self.LayerClass):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        return x


class InnerProductDecoder(nn.Module):
    def __init__(self):
        super(InnerProductDecoder, self).__init__()
        self.reset_parameters()

    def reset_parameters(self):
        pass

    @staticmethod
    def forward(embedding) -> Tensor:
        hat_adj = torch.sigmoid(torch.matmul(embedding, embedding.t()))
        return hat_adj
