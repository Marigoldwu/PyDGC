# -*- coding: utf-8 -*-
import argparse


def parse_arguments(arg_config: dict) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for arg_name, arg_default in arg_config.items():
        arg_type = type(arg_default)
        if arg_type == bool:
            parser.add_argument(f'-{arg_name}', dest=arg_name, action='store_true', default=False)
        else:
            parser.add_argument(f'--{arg_name}', type=arg_type, default=arg_default)
    args = parser.parse_args()
    args.dataset_name = args.dataset_name.upper()
    return args
