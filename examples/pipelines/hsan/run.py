# -*- coding: utf-8 -*-
from pydgc.pipelines import HSANPipeline
from pydgc.utils import parse_arguments

datasets = ["UAT"]

for datasets in datasets:
    args_default = {
        'cfg_file_path': 'config.yaml',
        'dataset_name': datasets
    }
    args = parse_arguments(args_default)

    pipeline = HSANPipeline(args)
    pipeline.run()
