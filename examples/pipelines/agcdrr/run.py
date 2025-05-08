# -*- coding: utf-8 -*-
from pydgc.pipelines import AGCDRRPipeline
from pydgc.utils import parse_arguments

datasets = ["WIKI", "CORA", "ACM", "CITE", "DBLP", "PUBMED", "BLOG", "FLICKR", "ROMAN", "USPS_3", "HHAR_3"]

for dataset in datasets:
    args_default = {
        'cfg_file_path': 'config.yaml',
        'dataset_name': dataset,
    }
    args = parse_arguments(args_default)

    pipeline = AGCDRRPipeline(args)
    pipeline.run()
