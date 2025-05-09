# -*- coding: utf-8 -*-
import os
import sys
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from pydgc.pipelines import DFCNPipeline
from pydgc.utils import parse_arguments

datasets = ["WIKI", "CORA", "ACM", "CITE", "DBLP", "BLOG", "FLICKR", "USPS_3", "HHAR_3", "PUBMED", "ROMAN", "ARXIV"]

for dataset in datasets:
    args = parse_arguments(dataset)
    pipeline = DFCNPipeline(args)
    pipeline.run()
