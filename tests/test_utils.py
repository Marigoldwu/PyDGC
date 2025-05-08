# -*- coding: utf-8 -*-
from pydgc.utils import generate_default_cfg

import unittest


class TestUtils(unittest.TestCase):
    def test_generate_default_cfg(self):
        generate_default_cfg(datasets="CORA")
        generate_default_cfg(datasets="CORA", save_path=".")
        generate_default_cfg(datasets="CITE", save_path="CITE")
        generate_default_cfg(datasets="ACM", save_path="ACM.yaml")
        generate_default_cfg(datasets="CORA", save_path="./config/single_dataset.yaml")
        generate_default_cfg(datasets=["ACM", "CITE", "CORA"], save_path="./config/multi_datasets.yaml")
