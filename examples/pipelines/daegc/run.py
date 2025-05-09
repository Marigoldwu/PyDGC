# -*- coding: utf-8 -*-
import os
import sys
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from pydgc.pipelines import DAEGCPipeline
from pydgc.utils import parse_arguments

datasets = ["WIKI", "CORA", "ACM", "CITE", "DBLP", "PUBMED", "ARXIV", "BLOG", "FLICKR", "ROMAN", "USPS_3", "HHAR_3"]

for dataset in datasets:
    args = parse_arguments(dataset)
    pipeline = DAEGCPipeline(args)
    pipeline.run()
