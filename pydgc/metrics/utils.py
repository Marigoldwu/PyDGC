# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.utils import to_dense_adj, contains_self_loops, add_self_loops

from pydgc.utils import Logger
from yacs.config import CfgNode as CN
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (accuracy_score, adjusted_rand_score, normalized_mutual_info_score,
                             homogeneity_score, completeness_score, f1_score, cluster, silhouette_score)


class DGCMetric:
    """DGC metric class.

    Args:
        ground_truth (np.array): Ground truth labels.
        predicted_labels (np.array): Predicted labels.
        embeddings (Tensor): Node embeddings.
        edge_index (Tensor): Edge index.
    """
    def __init__(self, ground_truth: np.ndarray, predicted_labels: np.ndarray, embeddings: Tensor, edge_index: Tensor):
        self.predicted_labels = predicted_labels
        self.ground_truth = ground_truth
        self.predicted_clusters = len(np.unique(self.predicted_labels))
        # self.n_clusters = len(np.unique(self.ground_truth))
        self.mapped_labels = None
        self.embeddings = embeddings
        self.edge_index = edge_index

    def accuracy(self, decimal: int = 4):
        """Calculate clustering accuracy after using the linear_sum_assignment function in SciPy to
        determine reassignments.

        Args:
            decimal (int, optional): The number of decimal places that need to be retained. Defaults to 4.

        Returns:
            float: Clustering accuracy.
        """
        if self.mapped_labels is not None:
            acc = accuracy_score(self.ground_truth, self.mapped_labels)
            return round(acc, decimal)
        n_clusters = max(len(np.unique(self.predicted_labels)), len(np.unique(self.ground_truth)))
        count_matrix = np.zeros((n_clusters, n_clusters), dtype=np.int64)
        for i in range(self.predicted_labels.size):
            count_matrix[self.predicted_labels[i], self.ground_truth[i]] += 1

        row_ind, col_ind = linear_sum_assignment(count_matrix.max() - count_matrix)
        reassignment = dict(zip(row_ind, col_ind))
        self.mapped_labels = np.vectorize(reassignment.get)(self.predicted_labels)
        acc = count_matrix[row_ind, col_ind].sum() / self.predicted_labels.size
        return round(acc, decimal)

    def f1_score(self, decimal: int = 4) -> float:
        """Calculate F1 score.

        Args:
            decimal (int, optional): The number of decimal places that need to be retained. Defaults to 4.

        Returns:
            float: F1 score.
        """
        if self.mapped_labels is not None:
            f1 = f1_score(self.ground_truth, self.mapped_labels, average='macro')
            return round(f1, decimal)
        n_clusters = max(len(np.unique(self.predicted_labels)), len(np.unique(self.ground_truth)))
        count_matrix = np.zeros((n_clusters, n_clusters), dtype=np.int64)
        for i in range(self.predicted_labels.size):
            count_matrix[self.predicted_labels[i], self.ground_truth[i]] += 1

        row_ind, col_ind = linear_sum_assignment(count_matrix.max() - count_matrix)
        reassignment = dict(zip(row_ind, col_ind))
        self.mapped_labels = np.vectorize(reassignment.get)(self.predicted_labels)
        f1 = f1_score(self.ground_truth, self.mapped_labels, average='macro')
        return round(f1, decimal)

    def nmi_score(self, decimal: int = 4) -> float:
        """Calculate NMI score.

        Args:
            decimal (int, optional): The number of decimal places that need to be retained. Defaults to 4.

        Returns:
            float: NMI score.
        """
        nmi = normalized_mutual_info_score(self.ground_truth, self.predicted_labels, average_method='arithmetic')
        return round(nmi, decimal)

    def ari_score(self, decimal: int = 4) -> float:
        """Calculate ARI score.

        Args:
            decimal (int, optional): The number of decimal places that need to be retained. Defaults to 4.

        Returns:
            float: ARI score.
        """
        ari = adjusted_rand_score(self.ground_truth, self.predicted_labels)
        return round(ari, decimal)

    def hom_score(self, decimal: int = 4) -> float:
        """Calculate homogeneity score.

        Args:
            decimal (int, optional): The number of decimal places that need to be retained. Defaults to 4.

        Returns:
            float: Homogeneity score.
        """
        hom = homogeneity_score(self.ground_truth, self.predicted_labels)
        return round(hom, decimal)

    def com_score(self, decimal: int = 4) -> float:
        """Calculate completeness score.

        Args:
            decimal (int, optional): The number of decimal places that need to be retained. Defaults to 4.

        Returns:
            float: Completeness score.
        """
        com = completeness_score(self.ground_truth, self.predicted_labels)
        return round(com, decimal)

    def sil_score(self, decimal: int = 4) -> float:
        """Calculate silhouette score.

        Args:
            decimal (int, optional): The number of decimal places that need to be retained. Defaults to 4.

        Returns:
            float: Silhouette score.
        """
        # if isinstance(self.embeddings, np.ndarray):
        #     embeddings = self.embeddings.copy()
        # else:
        embeddings = self.embeddings.clone()
        if embeddings.device != torch.device('cpu'):
            embeddings = embeddings.detach().cpu().numpy()
        if isinstance(embeddings, Tensor):
            embeddings = embeddings.numpy()
        sil = silhouette_score(embeddings, self.predicted_labels)
        return round(sil, decimal)

    def graph_reconstruction_error(self, decimal: int = 4) -> float:
        """Calculate graph reconstruction error.

        Args:
            decimal (int, optional): The number of decimal places that need to be retained. Defaults to 4.

        Returns:
            float: Graph reconstruction error.
        """
        if isinstance(self.embeddings, np.ndarray):
            self.embeddings = torch.from_numpy(self.embeddings)
        reconstructed = F.sigmoid(self.embeddings @ self.embeddings.t())
        if not contains_self_loops(self.edge_index):
            self.edge_index = add_self_loops(self.edge_index)[0]
        dense_adj = to_dense_adj(self.edge_index)[0]
        if reconstructed.device != torch.device('cpu'):
            reconstructed = reconstructed.detach().cpu()
        if dense_adj.device != torch.device('cpu'):
            dense_adj = dense_adj.detach().cpu()
        gre = F.mse_loss(reconstructed, dense_adj).item()
        return round(gre, decimal)

    def purity(self, decimal: int = 4) -> float:
        """Calculate purity score.

        Args:
            decimal (int, optional): The number of decimal places that need to be retained. Defaults to 4.

        Returns:
            float: Purity score.
        """
        contingency_matrix = cluster.contingency_matrix(self.ground_truth, self.predicted_labels)
        pur = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
        return round(pur, decimal)

    def evaluate_one_epoch(self,
                           logger: Logger,
                           cfg: CN = None) -> dict:
        """Evaluate one epoch.

        Args:
            logger (Logger): Logger.
            cfg (CN, optional): Config. Defaults to None.

        Returns:
            dict: Results with metric names as keys and metric values as values.
        """
        results = {}
        if cfg is None:
            results['ACC'] = self.accuracy()
            results['NMI'] = self.accuracy()
        else:
            if hasattr(cfg, 'evaluate'):
                cfg = cfg.evaluate
            elif hasattr(cfg, 'acc') or hasattr(cfg, 'nmi') or hasattr(cfg, 'ari') or hasattr(cfg, 'f1') or hasattr(cfg, 'hom') or hasattr(cfg, 'com') or hasattr(cfg, 'pur') or hasattr(cfg, 'sc') or hasattr(cfg, 'gre'):
                cfg = cfg
            if cfg.acc:
                results['ACC'] = self.accuracy()
            if cfg.nmi:
                results['NMI'] = self.nmi_score()
            if cfg.ari:
                results['ARI'] = self.ari_score()
            if cfg.f1:
                results['F1'] = self.f1_score()
            if cfg.hom:
                results['HOM'] = self.hom_score()
            if cfg.com:
                results['COM'] = self.com_score()
            if cfg.pur:
                results['PUR'] = self.purity()
            if cfg.sc:
                if self.predicted_clusters == 1:
                    results['SC'] = 0
                else:
                    results['SC'] = self.sil_score()
            if cfg.gre:
                results['GRE'] = self.graph_reconstruction_error()
        logger.info(results)
        return results


def build_results_dict(cfg: CN) -> dict:
    """Build results dict.

    Args:
        cfg (CN): Config.

    Returns:
        dict: Results dict.
    """
    results = {}
    for key, value in zip(cfg.keys(), cfg.values()):
        if key == 'each':
            continue
        if value:
            results[key.upper()] = []
    return results
