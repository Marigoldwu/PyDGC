# -*- coding: utf-8 -*-
import torch
import random
import numpy as np


def setup_seed(seed):
    """Fix the random seed.

    Args:
        seed (int): The random seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
