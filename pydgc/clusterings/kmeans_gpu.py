# -*- coding: utf-8 -*-
import torch
import numpy as np

from torch import Tensor
from typing import Tuple


class KMeansGPU:
    def __init__(self,
                 n_clusters: int,
                 *,
                 distance: str = 'euclidean',
                 tol: float = 1e-4,
                 max_iter: int = 500,
                 device: str = 'cuda'):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.distance = distance
        self.device = torch.device(device)
        self.cluster_centers_ = None
        self.labels_ = None

    def initialize(self, X: Tensor) -> np.array:
        """
        initialize cluster centers
        :param X: (torch.tensor) matrix
        :return: (np.array) initial state
        """
        num_samples = len(X)
        indices = np.random.choice(num_samples, self.n_clusters, replace=False)
        initial_state = X[indices]
        return initial_state

    def pairwise_distance(self, data1: Tensor, data2: Tensor) -> Tensor:
        # transfer to device
        data1, data2 = data1.to(self.device), data2.to(self.device)

        # N*1*M
        A = data1.unsqueeze(dim=1)

        # 1*N*M
        B = data2.unsqueeze(dim=0)

        dis = (A - B) ** 2.0
        # return N*N matrix for pairwise distance
        dis = dis.sum(dim=-1).squeeze()
        return dis

    def pairwise_cosine(self, data1: Tensor, data2: Tensor) -> Tensor:
        # transfer to device
        data1, data2 = data1.to(self.device), data2.to(self.device)

        # N*1*M
        A = data1.unsqueeze(dim=1)

        # 1*N*M
        B = data2.unsqueeze(dim=0)

        # normalize the points  | [0.3, 0.4] -> [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
        A_normalized = A / A.norm(dim=-1, keepdim=True)
        B_normalized = B / B.norm(dim=-1, keepdim=True)

        cosine = A_normalized * B_normalized

        # return N*N matrix for pairwise distance
        cosine_dis = 1 - cosine.sum(dim=-1).squeeze()
        return cosine_dis

    def fit(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        """
        perform kmeans
        :param X: (torch.tensor) matrix
        :return: (torch.tensor, torch.tensor) cluster ids, cluster centers
        """
        if self.distance == 'euclidean':
            pairwise_distance_function = self.pairwise_distance
        elif self.distance == 'cosine':
            pairwise_distance_function = self.pairwise_cosine
        else:
            raise NotImplementedError

        # convert to float
        X = X.float().to(self.device)

        # initialize
        dis_min = float('inf')
        initial_state_best = None
        for i in range(20):
            initial_state = self.initialize(X)
            dis = pairwise_distance_function(X, initial_state).sum()
            if dis < dis_min:
                dis_min = dis
                initial_state_best = initial_state

        self.cluster_centers_ = initial_state_best
        iteration = 0
        while iteration <= self.max_iter:
            dis = pairwise_distance_function(X, self.cluster_centers_)

            self.labels_ = torch.argmin(dis, dim=1)

            initial_state_pre = self.cluster_centers_.clone()

            for index in range(self.n_clusters):
                selected = torch.nonzero(self.labels_ == index).squeeze().to(self.device)

                selected = torch.index_select(X, 0, selected)
                self.cluster_centers_[index] = selected.mean(dim=0)

            center_shift = torch.sum(
                torch.sqrt(
                    torch.sum((self.cluster_centers_ - initial_state_pre) ** 2, dim=1)
                ))

            # increment iteration
            iteration = iteration + 1
            if center_shift ** 2 < self.tol:
                break

        return self.labels_.cpu(), self.cluster_centers_

    def predict(self, X: Tensor) -> Tensor:
        """
        predict using cluster centers
        :param X: (torch.tensor) matrix
        :return: (torch.tensor) cluster ids
        """
        if self.distance == 'euclidean':
            pairwise_distance_function = self.pairwise_distance
        elif self.distance == 'cosine':
            pairwise_distance_function = self.pairwise_cosine
        else:
            raise NotImplementedError

        # convert to float
        X = X.float().to(self.device)

        dis = pairwise_distance_function(X, self.cluster_centers_)
        self.labels_ = torch.argmin(dis, dim=1)

        return self.labels_.cpu()
