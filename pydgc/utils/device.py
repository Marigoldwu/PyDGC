# -*- coding: utf-8 -*-
"""
@Reference: https://github.com/snap-stanford/GraphGym/blob/master/graphgym/utils/device.py
"""
import os
import torch
import subprocess
import numpy as np


from yacs.config import CfgNode as CN


def count_parameters(model):
    """
    Count the parameters' number of the input model
    Note: The unit of return value is millions(M) if exceeds 1,000,000.

    :param model: the model instance you want to count
    :return: The number of model parameters, in Million (M).
    """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return round(num_params / 1e6, 6)


def get_gpu_memory_map():
    """
    Get the current gpu usage.
    """
    result = subprocess.check_output([
        'nvidia-smi', '--query-gpu=memory.used',
        '--format=csv,nounits,noheader'
    ], encoding='utf-8')
    gpu_memory = np.array([int(x) for x in result.strip().split('\n')])
    return gpu_memory


def get_current_gpu_usage(gpu_mem, device: str):
    """
    Get the current GPU memory usage.
    """
    if gpu_mem and device != 'cpu' and torch.cuda.is_available():
        result = subprocess.check_output([
            'nvidia-smi', '--query-compute-apps=pid,used_memory',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
        current_pid = os.getpid()
        used_memory = 0
        for line in result.strip().split('\n'):
            line = line.split(', ')
            if current_pid == int(line[0]):
                used_memory += int(line[1])
        return used_memory
    else:
        return -1


def auto_select_device(logger,
                       cfg: CN,
                       memory_max: int = 8000,
                       memory_bias: int = 200,
                       strategy: str = 'random'):
    """
    Auto select device for the experiment. Useful when having multiple GPUs.
    :param logger:
    :param cfg:
    :param memory_max: Threshold of existing GPU memory usage. GPUs with
    memory usage beyond this threshold will be deprioritized.
    :param memory_bias: A bias GPU memory usage added to all the GPUs.
        Avoid divided by zero error.
    :param strategy: 'random' (random select GPU) or 'greedy'
        (greedily select GPU)

    """
    if cfg.device != 'cpu' and torch.cuda.is_available():
        if cfg.device == 'auto':
            memory_raw = get_gpu_memory_map()
            if strategy == 'greedy' or np.all(memory_raw > memory_max):
                cuda = np.argmin(memory_raw)
                logger.info('GPU Mem: {}'.format(memory_raw))
                logger.info(
                    'Greedy select GPU, select GPU {} with mem: {}'.format(
                        cuda, memory_raw[cuda]))
            else:
                memory = 1 / (memory_raw + memory_bias)
                memory[memory_raw > memory_max] = 0
                gpu_prob = memory / memory.sum()
                cuda = np.random.choice(len(gpu_prob), p=gpu_prob)
                logger.info('GPU Mem: {}'.format(memory_raw))
                logger.info('GPU Prob: {}'.format(gpu_prob.round(2)))
                logger.info(
                    'Random select GPU, select GPU {} with mem: {}'.format(
                        cuda, memory_raw[cuda]))

            cfg.device = 'cuda:{}'.format(cuda)
    else:
        cfg.device = 'cpu'
    return cfg
