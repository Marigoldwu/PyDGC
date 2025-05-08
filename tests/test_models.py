# -*- coding: utf-8 -*-
from unittest import TestCase

from pydgc.models import DFCN
from yacs.config import CfgNode as CN

from pydgc.models.dfcn import IGAE


class TestModels(TestCase):
    def test_dfcn(self):
        cfg = CN()
        cfg.igae = CN()
        cfg.igae.dir = "./pretrain/ACM/"
        cfg.max_epoch = 30
        cfg.lr = 1e-3
        cfg.gamma_value = 0.1
        model = IGAE('1', 128, 256, 10, 10, 256, 128, 1000)
        print(list(model.parameters()))

