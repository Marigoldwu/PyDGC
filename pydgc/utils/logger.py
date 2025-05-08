# -*- coding: utf-8 -*-
import os
import time
import logging
import numpy as np

from rich.table import Table
from torch import Tensor

from . import count_parameters
from rich.console import Console
from rich.logging import RichHandler


def get_formatted_time():
    current_time = time.localtime()
    time_format = "%Y-%m-%d %H-%M-%S"
    formatted_time = time.strftime(time_format, current_time)
    return formatted_time


def create_logger(logger_name, log_mode='both', log_file_path=None, encoding='utf-8'):
    """
    Set up printing options

    :param logger_name: used to name logger.
    :param log_file_path: If print output to file, you must specify file path.
    :param log_mode: Print mode. Options: [file, stdout, both].
    :param encoding: Encoding mode, 'utf-8' for default.
    :
    """
    if log_mode != 'stdout' and log_file_path is None:
        raise ValueError("log_file_path must be specified when print output to log file!")
    logging.root.handlers = []
    logging_cfg = {'level': logging.INFO, 'format': '%(message)s'}
    h_stdout = RichHandler(show_path=False,
                           keywords=["Random seed",
                                     "Round", "Epoch", "Loss",
                                     "ACC", "NMI", "ARI", "F1", "HOM", "COM", "PUR",
                                     "Time cost"])
    dir_ = os.path.dirname(log_file_path)
    if not os.path.exists(dir_):
        os.makedirs(dir_)
    if log_mode == 'file':
        h_file = logging.FileHandler(log_file_path, encoding=encoding)
        logging_cfg['handlers'] = [h_file]
    elif log_mode == 'stdout':
        logging_cfg['handlers'] = [h_stdout]
    elif log_mode == 'both':
        h_file = logging.FileHandler(log_file_path, encoding=encoding)
        logging_cfg['handlers'] = [h_file, h_stdout]
    else:
        raise ValueError('Print option not supported, options: file, stdout, both')
    logging.basicConfig(**logging_cfg)
    return Logger(name=logger_name)


class Logger(object):
    def __init__(self, name):
        self.logger = logging.getLogger(name)

    def info(self, message):
        self.logger.info(message)

    def error(self, message):
        self.logger.error(message)

    def debug(self, message):
        self.logger.debug(message)

    def warning(self, message):
        self.logger.warning(message)

    def flag(self, message):
        self.logger.info(f"{'*' * 40}{message}{'*' * 40}")

    @staticmethod
    def table(results_dir: str, dataset_name: str, results_dict: dict, decimal: int = 4):
        table = Table(title=f"Clustering Results on Dataset {dataset_name}")
        if type(next(iter(results_dict.values()))) in [float, int, np.float32, np.float64, np.int32, np.int64]:
            table.add_column("Metric", justify="right", style="cyan", no_wrap=True)
            table.add_column("Value", justify="right", style="green", no_wrap=True)
            for key, value in results_dict.items():
                table.add_row(key, str(round(value, decimal)))
        else:
            table.add_column("Metric", justify="right", style="cyan", no_wrap=True)
            rounds = len(next(iter(results_dict.values())))
            for i in range(rounds):
                table.add_column(f"{i + 1}", justify="right", style="green", no_wrap=True)
            table.add_column("Avg.", justify="right", style="green", no_wrap=True)
            table.add_column("Std.", justify="right", style="green", no_wrap=True)
            for key, values in results_dict.items():
                table.add_row(key, *[str(round(value, decimal)) for value in values],
                              str(round(np.mean(values), decimal)),
                              str(round(np.std(values), decimal)))
        with open(os.path.join(results_dir, "results.txt"), "a+") as report_file:
            console = Console(file=report_file)
            console.print(get_formatted_time())
            console.print(table)
        console = Console()
        console.print(table)

    def loss(self, epoch, loss, decimal: int = 6):
        if isinstance(loss, Tensor):
            loss = loss.item()
        self.logger.info(f"Epoch: {epoch:0>4d}, Loss: {round(loss, decimal):0>.{decimal}f}")

    def model_info(self, model):
        self.logger.info(model)
        self.logger.info(f"Parameters: {count_parameters(model)} MB")
