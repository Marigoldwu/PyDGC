# -*- coding: utf-8 -*-
from argparse import Namespace
from ..models import S3GC


from . import BasePipeline
from ..utils import perturb_data


class S3GCPipeline(BasePipeline):
    def __init__(self, args: Namespace):
        super().__init__(args)

    def augment_data(self):
        """Data augmentation"""
        self.data = perturb_data(self.data, self.cfg.dataset.augmentation)

    def build_model(self):
        model = S3GC(self.logger, self.cfg)
        self.logger.model_info(model)
        return model
