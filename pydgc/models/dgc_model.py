# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from torch import Tensor
from ..utils import Logger
from typing import Tuple, Any, List, Dict
from abc import ABC, abstractmethod
from yacs.config import CfgNode as CN


class DGCModel(nn.Module, ABC):
    def __init__(self, logger: Logger, cfg: CN):
        super(DGCModel, self).__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.logger = logger

    @abstractmethod
    def reset_parameters(self):
        pass

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def loss(self, *args, **kwargs) -> Tensor:
        pass

    @abstractmethod
    def train_model(self, *args, **kwargs) -> Tuple[List, Tensor, Tensor, Dict]:
        pass

    @abstractmethod
    def get_embedding(self, *args, **kwargs) -> Tensor:
        pass

    @abstractmethod
    def clustering(self, *args, **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
        pass

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        pass
