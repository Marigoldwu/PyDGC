# -*- coding: utf-8 -*-
import os
import sys
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from pydgc.pipelines import SDCNPipeline
from pydgc.utils import parse_arguments

datasets = ["CORA", "BLOG"]

for drop_edge_prob in [0.2, 0.5, 0.8]:
    for dataset in datasets:
        args = parse_arguments(dataset)
        args.drop_edge = drop_edge_prob
        pipeline = SDCNPipeline(args)
        pipeline.run()

for drop_feature_prob in [0.2, 0.5, 0.8]:
    for dataset in datasets:
        args = parse_arguments(dataset)
        args.drop_feature = drop_feature_prob
        pipeline = SDCNPipeline(args)
        pipeline.run()

for add_edge_prob in [0.2, 0.5, 0.8]:
    for dataset in datasets:
        args = parse_arguments(dataset)
        args.add_edge = add_edge_prob
        pipeline = SDCNPipeline(args)
        pipeline.run()

for add_noise_prob in [0.1, 1, 10]:
    for dataset in datasets:
        args = parse_arguments(dataset)
        args.add_noise = add_noise_prob
        pipeline = SDCNPipeline(args)
        pipeline.run()
