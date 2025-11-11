# -*- coding: utf-8 -*-
import os
import sys
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from pydgc.pipelines import NS4GCPipeline
from pydgc.utils import parse_arguments

# datasets = ["WIKI", "CORA", "ACM", "CITE", "DBLP", "BLOG", "FLICKR", "USPS_3", "HHAR_3", "PUBMED", "ROMAN", "ARXIV"]
# datasets = ["CORA"]
datasets = ["UAT"]
for dataset in datasets:
    args = parse_arguments(dataset)
    pipeline = NS4GCPipeline(args)
    pipeline.run()
