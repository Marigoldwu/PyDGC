# -*- coding: utf-8 -*-
import argparse

ARGS_DEFAULT = {
        'cfg_file_path': 'config.yaml',
        'dataset_name': "ACM",
        'drop_edge': 0.0,
        'drop_feature': 0.0,
        'add_edge': 0.0,
        'add_noise': 0.0
    }


def parse_arguments(dataset_name: str = "ACM", arg_config: dict = None) -> argparse.Namespace:
    """
    增加命令行参数以便在命令行修改需要频繁修改的参数

    :param dataset_name: dataset name
    :param arg_config: 自定义参数
    :return: argparse.Namespace
    """
    ARGS_DEFAULT['dataset_name'] = dataset_name
    if arg_config is not None:
        ARGS_DEFAULT.update(arg_config)
    parser = argparse.ArgumentParser()
    for arg_name, arg_default in ARGS_DEFAULT.items():
        arg_type = type(arg_default)
        if arg_type == bool:
            parser.add_argument(f'-{arg_name}', dest=arg_name, action='store_true', default=False)
        else:
            parser.add_argument(f'--{arg_name}', type=arg_type, default=arg_default)
    args = parser.parse_args()
    args.dataset_name = args.dataset_name.upper()
    return args
