# -*- coding: utf-8 -*-
from pydgc.pipelines import SDCNPipeline
from pydgc.utils import parse_arguments

datasets = ["ACM"]

for datasets in datasets:
    args_default = {
        'cfg_file_path': 'config.yaml',
        'dataset_name': datasets,
        'pretrain': False,
        'flag': "PRETRAIN AE FOR SDCN"
    }
    args = parse_arguments(args_default)

    pipeline = SDCNPipeline(args)
    pipeline.run(pretrain=args.pretrain, flag=args.flag)
