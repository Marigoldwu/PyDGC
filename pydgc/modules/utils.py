# -*- coding: utf-8 -*-
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, SGConv


_layer_registry = {
    'linear': nn.Linear,
    'gcn': GCNConv,
    'gat': GATConv,
    'sage': SAGEConv,
    'sg': SGConv
}


class LayerRegistry:
    _registry = _layer_registry

    @classmethod
    def list_layers(cls):
        return list(cls._registry.keys())

    @classmethod
    def get_layer(cls, name):
        return cls._registry.get(name)


def register_layer(name: str, layer_class: nn.Module):
    """Register decorators/functions for custom layer types"""
    def decorator(cls):
        _layer_registry[name] = layer_class
        return cls
    return decorator
