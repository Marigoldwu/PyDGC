# -*- coding: utf-8 -*-
import time
import os.path as osp
import traceback
from argparse import Namespace

import torch
import numpy as np
from torch_geometric.utils import add_self_loops

from yacs.config import CfgNode as CN

from ..models import DGCModel
from abc import ABC, abstractmethod
from ..datasets import load_dataset
from ..utils.logger import create_logger
from ..utils.visualization import DGCVisual
from ..utils.device import auto_select_device
from ..metrics import build_results_dict
from ..utils import load_dataset_specific_cfg, setup_seed, get_formatted_time, dump_cfg, check_required_cfg


class BasePipeline(ABC):
    def __init__(self, args: Namespace):
        """
        Standardized pipeline for deep graph clustering

        :param args: command line arguments for setting values frequently changed
        """
        torch.set_default_dtype(torch.float32)
        self.args = args
        self.cfg_file_path = "config.yaml" if not hasattr(args, "cfg_file_path") else args.cfg_file_path
        if hasattr(args, "dataset_name"):
            self.dataset_name = args.dataset_name
        else:
            raise ValueError("Please specify dataset name! You can specify it in run.py or use --dataset_name!")
        self.cfg = None
        self.logger = None
        self.device = None
        self.data = None
        self.ground_truth = None
        self.predicted_labels = None
        self.results = {}
        self.loss_curve = []
        self.embeddings = None
        self.times = []
        self.current_round = 0

    def load_config(self):
        """load config from yaml"""
        self.cfg = load_dataset_specific_cfg(self.cfg_file_path, self.dataset_name)
        cfg = check_required_cfg(self.cfg, dataset_name=self.dataset_name)
        if isinstance(cfg, CN):
            self.cfg = cfg
        self.cfg.dataset.name = self.dataset_name
        if hasattr(self.args, 'drop_edge'):
            self.cfg.dataset.augmentation.drop_edge = float(self.args.drop_edge)
        if hasattr(self.args, 'drop_feature'):
            self.cfg.dataset.augmentation.drop_feature = float(self.args.drop_feature)
        if hasattr(self.args, 'add_edge'):
            self.cfg.dataset.augmentation.add_edge = float(self.args.add_edge)
        if hasattr(self.args, 'add_noise'):
            self.cfg.dataset.augmentation.add_noise = float(self.args.add_noise)
        self.cfg.evaluate.each = self.args.eval_each

    def load_logger(self):
        log_file = osp.join(self.cfg.logger.dir, f'{get_formatted_time()}.log')
        self.logger = create_logger(self.cfg.logger.name, self.cfg.logger.mode, log_file)
        auto_select_device(self.logger, self.cfg)
        self.device = torch.device(self.cfg.device)
        if self.cfg.train.rounds > 1:
            self.results = build_results_dict(self.cfg.evaluate)

    def load_dataset(self):
        try:
            if not self.cfg:
                raise ValueError("Please load config before loading data!")
            dataset = load_dataset(self.cfg.dataset.dir, self.dataset_name)
            self.cfg.dataset.n_clusters = dataset.num_classes
            if self.dataset_name.lower() == "arxiv":
                data = dataset
            else:
                data = dataset[0]
            data_self_loops = add_self_loops(data.edge_index)[0]
            data_self_loops = data_self_loops.to_dense()
            if data.x.is_sparse_csr:
                data.x = data.x.dense()
            data.x = data.x.float()
            self.cfg.dataset.num_nodes = data.num_nodes
            self.cfg.dataset.num_features = data.num_features
            num_edges = int((data.edge_index.shape[1]) / 2)
            self.cfg.dataset.num_edges = num_edges
            self.ground_truth = data.y.numpy()
            self.data = data
        except ValueError as e:
            print(e)
        except Exception as e:
            print(e)

    @abstractmethod
    def augment_data(self):
        pass

    @abstractmethod
    def build_model(self) -> DGCModel:
        """模型构建逻辑"""
        pass

    def evaluate(self, results):
        # cfg = self.cfg.evaluate
        # metric = DGCMetric(self.ground_truth, self.predicted_labels, self.embeddings, self.data.edge_index)
        # results = metric.evaluate_one_epoch(self.logger,
        #                                     acc=cfg.acc,
        #                                     nmi=cfg.nmi,
        #                                     ari=cfg.ari,
        #                                     f1=cfg.f1,
        #                                     sc=cfg.sc,
        #                                     hom=cfg.hom,
        #                                     com=cfg.com,
        #                                     pur=cfg.pur,
        #                                     gre=cfg.gre)
        if self.cfg.train.rounds > 1:
            for key, value in results.items():
                self.results[key].append(value)
        else:
            self.results = results

    def visualize(self):
        cfg = self.cfg.visualize
        plot = DGCVisual(save_path=cfg.dir, font_family=['Times New Roman', 'SimSun'], font_size=16)
        if cfg.tsne:
            self.logger.flag(f"TSNE START")
            plot.plot_clustering(self.embeddings.cpu().numpy(), self.predicted_labels, palette='Set2', method='tsne', filename='tsne_plot')
            self.logger.flag(f"TSNE END")
        if cfg.umap:
            self.logger.flag(f"UMAP START")
            plot.plot_clustering(self.embeddings.cpu().numpy(), self.predicted_labels, palette='Set2', method='umap', filename='umap_plot')
            self.logger.flag(f"UMAP END")
        if cfg.heatmap:
            self.logger.flag(f"HEATMAP START")
            plot.plot_heatmap(self.embeddings.cpu().numpy(), self.predicted_labels, method='inner_product', show_axis=False, show_color_bar=False)
            self.logger.flag(f"HEATMAP END")
        if cfg.loss:
            self.logger.flag(f"LOSS START")
            plot.plot_loss(self.loss_curve)
            self.logger.flag(f"LOSS END")

    def run(self, pretrain=False, flag="TRAIN"):
        try:
            self.load_config()
            self.load_logger()
            self.load_dataset()
            self.augment_data()
            if self.cfg.train.seed == -1:
                # set seed to no. current round
                for round_ in range(self.cfg.train.rounds):
                    self.logger.flag(f"Round: {round_+1}/{self.cfg.train.rounds} Dataset: {self.dataset_name}")
                    setup_seed(round_)
                    start = time.time()

                    model = self.build_model()
                    if pretrain:
                        if hasattr(model, 'pretrain'):
                            self.loss_curve = model.pretrain(self.data, self.cfg.train.pretrain, flag)
                            end = time.time()
                            time_cost = round(end - start, 4)
                            self.times.append(time_cost)
                            self.logger.info(f"Time cost: {time_cost}")
                            return
                        else:
                            raise ValueError("Model does not support pretraining!")
                    else:
                        self.loss_curve, embeddings, predicted_labels, results = model.train_model(self.data, self.cfg.train)
                        end = time.time()
                        time_cost = round(end - start, 4)
                        self.times.append(time_cost)
                        self.logger.info(f"Time cost: {time_cost}")

                        self.predicted_labels = predicted_labels.numpy()
                        self.embeddings = embeddings.detach()
                        self.evaluate(results)
                        if self.cfg.visualize.when == 'each':
                            self.visualize()
            else:
                # fixed seed with given seed
                setup_seed(self.cfg.train.seed)
                for round_ in range(self.cfg.train.rounds):
                    self.logger.flag(f"Round: {round_+1}/{self.cfg.train.rounds} Dataset: {self.dataset_name}")
                    start = time.time()

                    model = self.build_model()
                    if pretrain:
                        if hasattr(model, 'pretrain'):
                            self.loss_curve = model.pretrain(self.data, self.cfg.train.pretrain, flag)
                            end = time.time()
                            time_cost = end - start
                            self.times.append(time_cost)
                            self.logger.info(f"Time cost: {time_cost}")
                            return
                        else:
                            raise ValueError("Model does not support pretraining!")
                    else:
                        self.loss_curve, embeddings, predicted_labels, results = model.train_model(self.data, self.cfg.train)

                        end = time.time()
                        time_cost = end - start
                        self.times.append(time_cost)
                        self.logger.info(f"Time cost: {time_cost}")

                        self.predicted_labels = predicted_labels.numpy()
                        self.embeddings = embeddings.detach()
                        self.evaluate(results)
                        if self.cfg.visualize.when == 'each':
                            self.visualize()
            self.logger.table(self.cfg.logger.dir, self.dataset_name, self.results)
            self.logger.info(f"Average time cost: {np.mean(self.times)}±{np.std(self.times)}")
            mem_used = torch.cuda.max_memory_allocated(device=self.device) / 1024 / 1024
            self.logger.info(f"The max memory allocated to model is: {mem_used:.2f} MB.")
            if self.cfg.visualize.when == 'end':
                self.visualize()
            dump_cfg(self.cfg)
        except Exception as e:
            self.logger.error(str(e))
            self.logger.error(traceback.format_exc())
