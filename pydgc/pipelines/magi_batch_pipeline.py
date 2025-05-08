# -*- coding: utf-8 -*-
import torch
from torch_sparse import SparseTensor

from ..models import MAGIBatch
from . import BasePipeline
from argparse import Namespace
from ..utils import perturb_data
from torch_geometric.utils import to_undirected, add_remaining_self_loops


def get_sim(batch, adj, wt=20, wl=3):
    rowptr, col, _ = adj.csr()
    batch_size = batch.shape[0]
    batch_repeat = batch.repeat(wt)
    rw = adj.random_walk(batch_repeat, wl)[:, 1:]

    if not isinstance(rw, torch.Tensor):
        rw = rw[0]
    rw = rw.t().reshape(-1, batch_size).t()

    row, col, val = [], [], []
    for i in range(batch.shape[0]):
        rw_nodes, rw_times = torch.unique(rw[i], return_counts=True)
        row += [batch[i].item()] * rw_nodes.shape[0]
        col += rw_nodes.tolist()
        val += rw_times.tolist()

    unique_nodes = list(set(row + col))
    subg2g = dict(zip(unique_nodes, list(range(len(unique_nodes)))))

    row = [subg2g[x] for x in row]
    col = [subg2g[x] for x in col]
    idx = torch.tensor([subg2g[x] for x in batch.tolist()])

    adj_ = SparseTensor(row=torch.LongTensor(row), col=torch.LongTensor(col), value=torch.tensor(val),
                        sparse_sizes=(len(unique_nodes), len(unique_nodes)))

    adj_batch, _ = adj_.saint_subgraph(idx)
    adj_batch = adj_batch.set_diag(0.)
    # src, dst = dict_r[idx[adj_batch.storage.row()[3].item()].item()], dict_r[idx[adj_batch.storage.col()[3].item()].item()]
    return batch, adj_batch


def get_mask(adj):
    batch_mean = adj.mean(dim=1)
    mean = batch_mean[torch.LongTensor(adj.storage.row())]
    mask = (adj.storage.value() - mean) > - 1e-10
    row, col, val = adj.storage.row()[mask], adj.storage.col()[
        mask], adj.storage.value()[mask]
    adj_ = SparseTensor(row=row, col=col, value=val,
                        sparse_sizes=(adj.size(0), adj.size(1)))
    return adj_


class MAGIBatchPipeline(BasePipeline):
    def __init__(self, args: Namespace):
        super().__init__(args.cfg_file_path, args.dataset_name)

    def augment_data(self):
        """Data augmentation"""
        self.data = perturb_data(self.data, self.cfg.dataset.augmentation)

        edge_index = to_undirected(add_remaining_self_loops(self.data.edge_index)[0])

        N, E, num_features = self.data.x.shape[0], edge_index.shape[-1], self.data.x.shape[-1]

        adj = SparseTensor(row=edge_index[0],
                           col=edge_index[1], sparse_sizes=(N, N))
        adj.fill_value_(1.)
        assert len(self.cfg.model.dims.encoder) == len(self.cfg.dataset.size)

        self.data.adj = adj

    def build_model(self):
        model = MAGIBatch(self.logger, self.cfg)
        self.logger.model_info(model)
        return model

