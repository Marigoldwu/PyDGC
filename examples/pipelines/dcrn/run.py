# -*- coding: utf-8 -*-
from pydgc.pipelines import DCRNPipeline
from pydgc.utils import parse_arguments

datasets = ["ACM"]

for datasets in datasets:
    args_default = {
        'cfg_file_path': 'config.yaml',
        'dataset_name': datasets,
    }
    args = parse_arguments(args_default)

    pipeline = DCRNPipeline(args)
    pipeline.run()
