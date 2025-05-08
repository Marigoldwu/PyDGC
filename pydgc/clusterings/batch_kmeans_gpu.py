# -*- coding: utf-8 -*-
"""
Reference: https://github.com/EdisonLeeeee/MAGI/blob/master/magi/batch_kmeans_cuda.py
"""
from functools import partial

import numpy as np
import torch
from tqdm import tqdm


def initialize(X, num_clusters, seed):
    """
    initialize cluster centers
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param seed: (int) seed for kmeans
    :return: (np.array) initial state
    """
    num_samples = len(X)
    if seed == None:
        indices = np.random.choice(num_samples, num_clusters, replace=False)
    else:
        np.random.seed(seed)
        indices = np.random.choice(num_samples, num_clusters, replace=False)
    initial_state = X[indices]
    return initial_state


def kmeans(
        X,
        num_clusters,
        distance='euclidean',
        batch_size=100000,
        cluster_centers=[],
        tol=1e-4,
        tqdm_flag=True,
        iter_limit=0,
        device=torch.device('cpu'),
        gamma_for_soft_dtw=0.001,
        seed=None,
):
    """
    perform kmeans
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param seed: (int) seed for kmeans
    :param tol: (float) threshold [default: 0.0001]
    :param device: (torch.device) device [default: cpu]
    :param tqdm_flag: Allows to turn logs on and off
    :param iter_limit: hard limit for max number of iterations
    :param gamma_for_soft_dtw: approaches to (hard) DTW as gamma -> 0
    :return: (torch.tensor, torch.tensor) cluster ids, cluster centers
    """
    if tqdm_flag:
        print(f'running k-means on {device}..')

    if distance == 'euclidean':
        pairwise_distance_function = partial(pairwise_distance, batch_size=batch_size, device=device, tqdm_flag=tqdm_flag)
    else:
        raise NotImplementedError

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)
    if type(cluster_centers) == list:
        initial_state = initialize(X, num_clusters, seed=seed)
    else:
        if tqdm_flag:
            print('resuming')
        # find data point closest to the initial cluster center
        initial_state = cluster_centers
        dis = pairwise_distance_function(X, initial_state)
        choice_points = torch.argmin(dis, dim=0)
        initial_state = X[choice_points]
        initial_state = initial_state.to(device)

    iteration = 0
    if tqdm_flag:
        tqdm_meter = tqdm(desc='[running kmeans]')
    while True:
        choice_cluster = pairwise_distance_function(X, initial_state)

        initial_state_pre = initial_state.clone()

        for index in range(num_clusters):
            # selected = idx[choice_cluster == index].to(device)
            # selected = X[selected]
            selected = torch.nonzero(choice_cluster == index).squeeze().to(device)
            selected = torch.index_select(X, 0, selected)


            # https://github.com/subhadarship/kmeans_pytorch/issues/16
            if selected.shape[0] == 0:
                selected = X[torch.randint(len(X), (1,))]

            initial_state[index] = selected.mean(dim=0)

        center_shift = torch.sum(
            torch.sqrt(
                torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
            ))

        # increment iteration
        iteration = iteration + 1

        # update tqdm meter
        if tqdm_flag:
            tqdm_meter.set_postfix(
                iteration=f'{iteration}',
                center_shift=f'{center_shift ** 2:0.6f}',
                tol=f'{tol:0.6f}'
            )
            tqdm_meter.update()
        if center_shift ** 2 < tol:
            break
        if iter_limit != 0 and iteration >= iter_limit:
            break

    return choice_cluster.cpu(), initial_state.cpu()


def kmeans_predict(
        X,
        cluster_centers,
        batch_size=100000,
        distance='euclidean',
        device=torch.device('cpu'),
        gamma_for_soft_dtw=0.001,
        tqdm_flag=True
):
    """
    predict using cluster centers
    :param X: (torch.tensor) matrix
    :param cluster_centers: (torch.tensor) cluster centers
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param device: (torch.device) device [default: 'cpu']
    :param gamma_for_soft_dtw: approaches to (hard) DTW as gamma -> 0
    :return: (torch.tensor) cluster ids
    """
    if tqdm_flag:
        print(f'predicting on {device}..')

    if distance == 'euclidean':
        pairwise_distance_function = partial(pairwise_distance, batch_size=batch_size, device=device, tqdm_flag=tqdm_flag)
    else:
        raise NotImplementedError

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)

    choice_cluster = pairwise_distance_function(X, cluster_centers, batch_size=batch_size)

    return choice_cluster.cpu()


def pairwise_distance(data1, data2, batch_size=100000, device=torch.device('cpu'), tqdm_flag=True):
    if tqdm_flag:
        print(f'device is :{device}')

    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)
    if batch_size == -1:
        # full batch kmeans
        dis_ = (A - B) ** 2.0
        # return N*N matrix for pairwise distance
        dis_ = dis_.sum(dim=-1).squeeze()
        return torch.argmin(dis_, dim=1)
    else:
        # mini-batch kmeans
        choice_cluster = torch.zeros(data1.shape[0])
        for batch_idx in tqdm(range(int(np.ceil(data1.shape[0] / batch_size)))):
            dis = (A[batch_idx * batch_size: (batch_idx + 1) * batch_size] - B) ** 2.0
            dis = dis.sum(dim=-1).squeeze()
            choice_ = torch.argmin(dis, dim=1)
            choice_cluster[batch_idx * batch_size: (batch_idx + 1) * batch_size] = choice_
        choice_cluster = choice_cluster.long()
        return choice_cluster
