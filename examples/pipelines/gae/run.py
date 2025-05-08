# -*- coding: utf-8 -*-
from pydgc.pipelines import GAEPipeline
from pydgc.utils import parse_arguments

datasets = ["WIKI", "CORA", "ACM", "CITE", "DBLP", "PUBMED", "ARXIV", "BLOG", "FLICKR", "ROMAN", "USPS_3", "HHAR_3"]

for dataset in datasets:
    args = parse_arguments(dataset)
    pipeline = GAEPipeline(args)
    pipeline.run()
