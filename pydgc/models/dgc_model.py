# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from torch import Tensor
from ..utils import Logger
from typing import Tuple, Any, List, Dict
from abc import ABC, abstractmethod
from yacs.config import CfgNode as CN


class DGCModel(nn.Module, ABC):
    """Deep Graph Clustering base Model.

    Implement abstractmethod reset_parameters, forward, loss, train_model, get_embedding, clustering, evaluate.

    Args:
        logger (Logger): Logger.
        cfg (CN): Configuration.

    Returns:
        torch.Tensor: Output features.
    """
    def __init__(self, logger: Logger, cfg: CN):
        super(DGCModel, self).__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.logger = logger

    @abstractmethod
    def reset_parameters(self):
        """Reset model parameters."""
        pass

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """Model forward pass."""
        pass

    @abstractmethod
    def loss(self, *args, **kwargs) -> Tensor:
        """Model loss function."""
        pass

    @abstractmethod
    def train_model(self, *args, **kwargs) -> Tuple[List, List, Tensor, Tensor, Dict]:
        """Model training function."""
        pass

    @abstractmethod
    def get_embedding(self, *args, **kwargs) -> Tensor:
        """Get model embedding."""
        pass

    @abstractmethod
    def clustering(self, *args, **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
        """Clustering function."""
        pass

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        """Model evaluation function."""
        pass
