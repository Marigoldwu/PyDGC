# -*- coding: utf-8 -*-
from pydgc.pipelines import MAGIPipeline
from pydgc.pipelines import MAGIBatchPipeline
from pydgc.utils import parse_arguments

datasets = ["ARXIV"]
FULL = ["wiki", "cora", "acm", "cite", "dblp", "pubmed", "blog", "flickr", "roman", "usps", "hhar"]
BATCH = ["arxiv"]
for dataset in datasets:
    args_default = {
        'cfg_file_path': 'config.yaml',
        'dataset_name': dataset,
    }
    args = parse_arguments(args_default)
    if dataset.lower() in BATCH:
        pipeline = MAGIBatchPipeline(args)
        pipeline.run()
    else:
        pipeline = MAGIPipeline(args)
        pipeline.run()
