# -*- coding: utf-8 -*-
import os
import sys
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from pydgc.pipelines import MAGIPipeline
from pydgc.utils import parse_arguments

datasets = ["WIKI", "CORA", "ACM", "CITE", "DBLP", "PUBMED", "ARXIV", "BLOG", "FLICKR", "ROMAN", "USPS_3", "HHAR_3"]

FULL = ["wiki", "cora", "acm", "cite", "dblp", "pubmed", "blog", "flickr", "roman", "usps", "hhar"]
BATCH = ["arxiv"]
for dataset in datasets:
    args = parse_arguments(dataset)
    if dataset.lower() in BATCH:
        pass
    else:
        pipeline = MAGIPipeline(args)
        pipeline.run()
