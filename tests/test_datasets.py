# -*- coding: utf-8 -*-
import unittest

from pydgc.datasets import DGCGraphDataset, DGCNonGraphDataset, load_dataset
from torch_geometric.utils import homophily


class TestDatasets(unittest.TestCase):
    def test_load_dataset(self):
        datasets = ['HHAR_3']
        for name in datasets:
            name = name.upper()
            dataset = load_dataset(dataset_dir=f'./data/HHAR', dataset_name=name)
            data = dataset[0]
            homo_ratio = homophily(data.edge_index, data.y, method='edge')
            print(f'{name} {data.num_nodes}, {data.num_edges}, {data.num_features}, {dataset.num_classes}, {round(homo_ratio, 2)}')
