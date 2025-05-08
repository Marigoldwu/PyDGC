# -*- coding: utf-8 -*-
import os
import yaml
from yacs.config import CfgNode as CN

AUGMENT = ["add_self_loops", "drop_edge", "drop_feature", "add_noise", "add_edge"]


REQUIRED_CONFIG = [
    "device",
    {
        "dataset": ["dir", "augmentation"]
    },
    {
        "logger": ["dir", "name", "mode"]
    },
    {
        "model": ["dims"]
    },
    {
        "train": ["rounds", "seed", "lr", "max_epoch"]
    },
    {
        "evaluate": ["each", "acc", "nmi", "ari", "f1", "hom", "com", "pur", "sc", "gre"]
    },
    {
        "visualize": ["tsne", "umap", "heatmap", "loss", "dir", "when"]
    }
]


def validate_and_create_path(save_path):
    """
    Validate whether save_path is valid or not.
    If it contains directory and is valid but not exists, create directory.

    :param save_path:
    :return:
    """
    if os.sep not in save_path and (os.altsep and os.altsep not in save_path):
        return False

    directory = os.path.dirname(save_path)

    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except PermissionError:
            return False
    return True


def default_cfg(dataset_name) -> CN:
    dataset_name = dataset_name.upper()
    _C = CN()
    _C.device = "cuda"
    _C.dataset = CN()
    _C.dataset.name = dataset_name
    _C.dataset.dir = os.path.join("../data/", dataset_name)
    _C.dataset.augmentation = CN()
    _C.dataset.augmentation.add_self_loops = True
    _C.dataset.augmentation.drop_edge = 0.0
    _C.dataset.augmentation.drop_feature = 0.0
    _C.dataset.augmentation.add_edge = 0.0
    _C.dataset.augmentation.add_noise = 0.0
    _C.logger = CN()
    _C.logger.dir = os.path.join("./log/", dataset_name)
    _C.logger.name = dataset_name
    _C.logger.mode = "both"
    _C.model = CN()
    _C.model.dims = [256, 16]
    _C.train = CN()
    _C.train.rounds = 100
    _C.train.seed = -1
    _C.train.lr = 0.001
    _C.train.max_epoch = 50
    _C.evaluate = CN()
    _C.evaluate.each = False
    _C.evaluate.acc = True
    _C.evaluate.nmi = True
    _C.evaluate.ari = True
    _C.evaluate.f1 = True
    _C.evaluate.hom = True
    _C.evaluate.com = True
    _C.evaluate.pur = True
    _C.evaluate.sc = True
    _C.evaluate.gre = True
    _C.visualize = CN()
    _C.visualize.tsne = False
    _C.visualize.umap = False
    _C.visualize.heatmap = False
    _C.visualize.loss = True
    _C.visualize.dir = os.path.join("./visualization/", dataset_name)
    _C.visualize.when = 'end'
    _C.freeze()
    return _C.clone()


def yaml_to_cfg(yaml_data):
    """
    transform YAML into CfgNode
    :param yaml_data: data loaded from yaml
    :return: transformed CfgNode
    """
    cfg = CN()
    for key, value in yaml_data.items():
        if isinstance(value, dict):
            cfg[key] = yaml_to_cfg(value)
        else:
            cfg[key] = value
    return cfg


def dump_cfg(cfg: CN, save_path=None):
    """
    Records the configuration of this experiment.
    :param cfg:
    :param save_path:
    :return:
    """
    if save_path is None:
        if hasattr(cfg, 'logger') and hasattr(cfg.logger, 'dir'):
            cfg_file = os.path.join(cfg.logger.dir, "config.yaml")
        else:
            cfg_file = "config.yaml"
    else:
        if not validate_and_create_path(save_path):
            cfg_file = "config.yaml"
        elif os.path.isdir(save_path):
            cfg_file = os.path.join(save_path, "config.yaml")
        else:
            cfg_file = save_path
    with open(cfg_file, 'w', encoding="utf-8") as f:
        cfg.dump(stream=f, default_flow_style=False, indent=4)


def load_dataset_specific_cfg(cfg_file_path, dataset_name):
    """
    load config on specified dataset
    :param cfg_file_path: path of config file
    :param dataset_name: name of specific dataset
    :return:
    """
    try:
        dataset_name = dataset_name.upper()
        with open(cfg_file_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
        cfg_all = yaml_to_cfg(yaml_data)
        cfg_keys = cfg_all.keys()
        if dataset_name not in cfg_keys:
            raise ValueError
        cfg = getattr(cfg_all, dataset_name)
        return cfg.clone()
    except FileNotFoundError:
        print(f"{cfg_file_path} does not exist!")
    except ValueError:
        print(f"Could not find dataset {dataset_name} in {cfg_file_path}.")
    except Exception as e:
        print(f"Unknown error occurred: {e}")
    return None


def check_required_cfg(cfg: CN, dataset_name, auto_complete=True):
    missing_cfg = []
    cfg_keys = cfg.keys()

    for item in REQUIRED_CONFIG:
        if isinstance(item, str):
            if item not in cfg_keys:
                missing_cfg.append(item)
        if isinstance(item, dict):
            for key, sub_keys in item.items():
                if key not in cfg.keys():
                    missing_cfg.append(key)
                    continue
                for sub_key in sub_keys:
                    if sub_key not in getattr(cfg, key).keys():
                        missing_cfg.append(f"{key}.{sub_key}")
                    if sub_key == "augmentation":
                        for aug in AUGMENT:
                            if aug not in getattr(getattr(cfg, key), sub_key).keys():
                                missing_cfg.append(f"{key}.{sub_key}.{aug}")

    if len(missing_cfg) == 0:
        return True

    if auto_complete:
        print(f"Missing config items: {missing_cfg}")
        complete_value = []
        default_ = default_cfg(dataset_name)
        for item in missing_cfg:
            if "." in item:
                if item.count(".") == 1:
                    key, sub_key = item.split(".")
                    cfg[key][sub_key] = getattr(default_, key)[sub_key]
                    complete_value.append(f"{item}: {cfg[key][sub_key]}")
                else:
                    key, sub_key, sub_sub_key = item.split(".")
                    cfg[key][sub_key][sub_sub_key] = getattr(default_, key)[sub_key][sub_sub_key]
                    complete_value.append(f"{item}: {cfg[key][sub_key][sub_sub_key]}")
            else:
                cfg[item] = getattr(default_, item)
                complete_value.append(f"{item}: {cfg[item]}")
        print(f"Complete missing config items: {complete_value}")
        return cfg
    else:
        raise ValueError(f"Missing config items: {missing_cfg}")


def generate_default_cfg(datasets: str or list, save_path=None):
    """
    generate default config
    :param datasets:
    :param save_path:
    :return:
    """
    root = CN()
    if isinstance(datasets, str):
        dataset_name = datasets
        cfg = default_cfg(dataset_name)
        setattr(root, dataset_name.upper(), cfg)
    else:
        for dataset_name in datasets:
            cfg = default_cfg(dataset_name)
            setattr(root, dataset_name.upper(), cfg)
    dump_cfg(root, save_path=save_path)
