# -*- coding: utf-8 -*-
from typing import Any

import torch.nn as nn
import torch.nn.functional as F
from ..clusterings import kmeans
from sklearn.cluster import SpectralClustering, KMeans
from torch_geometric.data import Data

from torch_geometric.nn import GCNConv

from . import DGCModel

from yacs.config import CfgNode as CN

from ..metrics import DGCMetric
from ..utils import Logger
import numpy as np

from typing import Callable, List, NamedTuple, Optional, Tuple, Union

import torch
from torch import Tensor
from torch_sparse import SparseTensor


class Encoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels, base_model=GCNConv, dropout: float = 0.5, ns: float = 0.5):
        super(Encoder, self).__init__()
        self.base_model = base_model
        self.dropout = dropout
        self.k = len(hidden_channels)
        self.ns = ns

        self.convs = nn.ModuleList()
        self.convs.extend([base_model(in_channels, hidden_channels[0])])

        for i in range(1, self.k):
            self.convs.extend(
                [base_model(hidden_channels[i - 1], hidden_channels[i])])

        self.reset_parameters()

    def reset_parameters(self):
        """初始化模型参数"""
        for i in range(self.k):
            self.convs[i].reset_parameters()

    def forward(self, x: torch.Tensor, edge_index=None, adjs=None, dropout=True):
        if not adjs:
            for i in range(self.k):
                if dropout:
                    x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.convs[i](x, edge_index)
                x = F.leaky_relu(x, self.ns)
        else:
            for i, (edge_index, _, size) in enumerate(adjs):
                if dropout:
                    x = F.dropout(x, p=self.dropout, training=self.training)
                x_target = x[:size[1]]  # Target nodes are always placed first.
                x = self.convs[i]((x, x_target), edge_index)
                x = F.leaky_relu(x, self.ns)
        return x


class Loss(nn.Module):
    def __init__(self, temperature=0.07, scale_by_temperature=True, scale_by_weight=False):
        super(Loss, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature
        self.scale_by_weight = scale_by_weight

    def forward(self, out, mask):
        device = (torch.device('cuda') if out.is_cuda else torch.device('cpu'))

        row, col, val = mask.storage.row(), mask.storage.col(), mask.storage.value()
        row, col, val = row.to(device), col.to(device), val.to(device)
        batch_size = out.shape[0]

        # compute logits
        dot = torch.matmul(out, out.T)
        dot = torch.div(dot, self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(dot, dim=1, keepdim=True)
        dot = dot - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones(batch_size, batch_size).to(device),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )

        exp_logits = torch.exp(dot) * logits_mask
        log_probs = dot - torch.log(exp_logits.sum(1, keepdim=True))

        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")

        labels = row.view(row.shape[0], 1)
        unique_labels, labels_count = labels.unique(dim=0, return_counts=True)
        log_probs = log_probs[row, col]

        log_probs = log_probs.view(-1, 1)
        loss = torch.zeros_like(unique_labels, dtype=torch.float).to(device)
        loss.scatter_add_(0, labels, log_probs)
        loss = -1 * loss / labels_count.float().unsqueeze(1)

        if self.scale_by_temperature:
            loss *= self.temperature
        loss = loss.mean()
        return loss


def clustering(feature, n_clusters, true_labels, kmeans_device='cpu', batch_size=100000, tol=1e-4,
               device=torch.device('cuda:0'), spectral_clustering=False):
    if spectral_clustering:
        if isinstance(feature, torch.Tensor):
            feature = feature.numpy()
        print("spectral clustering on cpu...")
        Cluster = SpectralClustering(
            n_clusters=n_clusters, affinity='precomputed', random_state=0)
        f_adj = np.matmul(feature, np.transpose(feature))
        predict_labels = Cluster.fit_predict(f_adj)
    else:
        if kmeans_device == 'cuda':
            if isinstance(feature, np.ndarray):
                feature = torch.tensor(feature)
            print("kmeans on gpu...")
            predict_labels, _ = kmeans(
                X=feature, num_clusters=n_clusters, batch_size=batch_size, tol=tol, device=device)
            predict_labels = predict_labels.numpy()
        else:
            if isinstance(feature, torch.Tensor):
                feature = feature.numpy()
            print("kmeans on cpu...")
            Cluster = KMeans(n_clusters=n_clusters, max_iter=10000, n_init=20)
            predict_labels = Cluster.fit_predict(feature)
    return torch.from_numpy(predict_labels), None


def scale(z: torch.Tensor):
    zmax = z.max(dim=1, keepdim=True)[0]
    zmin = z.min(dim=1, keepdim=True)[0]
    z_std = (z - zmin) / ((zmax - zmin) + 1e-20)
    z_scaled = z_std
    return z_scaled


class EdgeIndex(NamedTuple):
    edge_index: Tensor
    e_id: Optional[Tensor]
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        edge_index = self.edge_index.to(*args, **kwargs)
        e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None else None
        return EdgeIndex(edge_index, e_id, self.size)


class Adj(NamedTuple):
    adj_t: SparseTensor
    e_id: Optional[Tensor]
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        adj_t = self.adj_t.to(*args, **kwargs)
        e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None else None
        return Adj(adj_t, e_id, self.size)


class NeighborSampler(torch.utils.data.DataLoader):
    r"""
    This code adapted from the pytorch geometric
    (https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/loader/neighbor_sampler.py).
    """

    def __init__(self,
                 edge_index: Union[Tensor, SparseTensor],
                 adj: SparseTensor,
                 sizes: List[int],
                 is_train: bool = False,
                 wt: int = 20,
                 wl: int = 4,
                 drop_last=False,
                 node_idx: Optional[Tensor] = None,
                 num_nodes: Optional[int] = None,
                 return_e_id: bool = True,
                 transform: Callable = None,
                 **kwargs):

        edge_index = edge_index.to('cpu')

        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']
        if 'dataset' in kwargs:
            del kwargs['dataset']

        # Save for Pytorch Lightning < 1.6:
        self.edge_index = edge_index
        self.adj = adj
        self.node_idx = node_idx
        self.num_nodes = num_nodes
        self.is_train = is_train
        self.drop_last = drop_last
        self.wt = wt
        self.wl = wl

        self.sizes = sizes
        self.return_e_id = return_e_id
        self.transform = transform
        self.is_sparse_tensor = isinstance(edge_index, SparseTensor)
        self.__val__ = None

        # Obtain a *transposed* `SparseTensor` instance.
        if not self.is_sparse_tensor:
            if (num_nodes is None and node_idx is not None
                    and node_idx.dtype == torch.bool):
                num_nodes = node_idx.size(0)
            if (num_nodes is None and node_idx is not None
                    and node_idx.dtype == torch.long):
                num_nodes = max(int(edge_index.max()), int(node_idx.max())) + 1
            if num_nodes is None:
                num_nodes = int(edge_index.max()) + 1

            value = torch.arange(edge_index.size(1)) if return_e_id else None
            self.adj_t = SparseTensor(row=edge_index[0], col=edge_index[1],
                                      value=value,
                                      sparse_sizes=(num_nodes, num_nodes)).t()
        else:
            adj_t = edge_index
            if return_e_id:
                self.__val__ = adj_t.storage.value()
                value = torch.arange(adj_t.nnz())
                adj_t = adj_t.set_value(value, layout='coo')
            self.adj_t = adj_t

        self.adj_t.storage.rowptr()

        if node_idx is None:
            node_idx = torch.arange(self.adj_t.sparse_size(0))
        elif node_idx.dtype == torch.bool:
            node_idx = node_idx.nonzero(as_tuple=False).view(-1)

        super().__init__(
            node_idx.view(-1).tolist(), collate_fn=self.sample, drop_last=self.drop_last, **kwargs)

    def get_batch(self, random_nodes):
        random_nodes_count = random_nodes.shape[0]
        rowptr, col, _ = self.adj.csr()

        # stage one
        random_nodes_repeat = random_nodes.repeat(self.wt)
        rw1 = self.adj.random_walk(random_nodes_repeat, self.wl)[:, 1:]
        if not isinstance(rw1, torch.Tensor):
            rw1 = rw1[0]
        rw1 = rw1.t().reshape(-1, random_nodes_count).t()
        batch = []
        for i in range(random_nodes_count):
            rw_nodes, rw_times = torch.unique(rw1[i], return_counts=True)
            nodes = rw_nodes[rw_times > rw_times.float().mean()].tolist()
            batch += nodes
        batch += random_nodes.tolist()
        batch = torch.tensor(batch).unique()

        # stage two
        batch_size = batch.shape[0]
        batch_repeat = batch.repeat(self.wt)
        rw2 = self.adj.random_walk(batch_repeat, self.wl)[:, 1:]
        if not isinstance(rw2, torch.Tensor):
            rw2 = rw2[0]
        rw2 = rw2.t().reshape(-1, batch_size).t()

        row, col, val = [], [], []
        for i in range(batch.shape[0]):
            rw2_nodes, rw2_times = torch.unique(rw2[i], return_counts=True)
            row += [batch[i].item()] * rw2_nodes.shape[0]
            col += rw2_nodes.tolist()
            val += rw2_times.tolist()

        unique_nodes = list(set(row + col))
        subg2g = dict(zip(unique_nodes, list(range(len(unique_nodes)))))

        row = [subg2g[x] for x in row]
        col = [subg2g[x] for x in col]
        idx = torch.tensor([subg2g[x] for x in batch.tolist()])

        adj_ = SparseTensor(row=torch.LongTensor(row), col=torch.LongTensor(col), value=torch.tensor(val),
                            sparse_sizes=(len(unique_nodes), len(unique_nodes)))

        adj_batch, _ = adj_.saint_subgraph(idx)

        # adj_batch = adj_batch.set_diag(0.)  # bug
        adj_batch_sp = adj_batch.to_scipy(layout='coo')
        adj_batch_sp.setdiag([0] * idx.shape[0])
        adj_batch = SparseTensor.from_scipy(adj_batch_sp)
        return batch, adj_batch

    def sample(self, batch):
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)
        adj_batch = None
        if self.is_train:
            batch, adj_batch = self.get_batch(batch)
        batch_size: int = len(batch)

        adjs = []
        n_id = batch
        for size in self.sizes:
            adj_t, n_id = self.adj_t.sample_adj(n_id, size, replace=False)
            e_id = adj_t.storage.value()
            size = adj_t.sparse_sizes()[::-1]
            if self.__val__ is not None:
                adj_t.set_value_(self.__val__[e_id], layout='coo')

            if self.is_sparse_tensor:
                adjs.append(Adj(adj_t, e_id, size))
            else:
                row, col, _ = adj_t.coo()
                edge_index = torch.stack([col, row], dim=0)
                adjs.append(EdgeIndex(edge_index, e_id, size))

        adjs = adjs[0] if len(adjs) == 1 else adjs[::-1]
        out = (batch_size, n_id, adjs)
        out = self.transform(*out) if self.transform is not None else out
        return out, adj_batch, batch

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(sizes={self.sizes})'


def get_mask(adj):
    batch_mean = adj.mean(dim=1)
    mean = batch_mean[torch.LongTensor(adj.storage.row())]
    mask = (adj.storage.value() - mean) > - 1e-10
    row, col, val = adj.storage.row()[mask], adj.storage.col()[
        mask], adj.storage.value()[mask]
    adj_ = SparseTensor(row=row, col=col, value=val,
                        sparse_sizes=(adj.size(0), adj.size(1)))
    return adj_


class MAGIBatch(DGCModel):

    def __init__(self, logger: Logger, cfg: CN):
        super(MAGIBatch, self).__init__(logger, cfg)
        encoder_dims = cfg.model.dims.encoder
        projection_dims = cfg.model.dims.projection
        encoder_dims.insert(0, cfg.dataset.num_features)
        self.encoder = Encoder(encoder_dims[0], encoder_dims[1:], base_model=GCNConv,
                               dropout=cfg.model.dropout, ns=cfg.model.ns).to(self.device)
        self.tau = cfg.model.tau
        self.in_channels = encoder_dims[-1]
        self.project_hidden = projection_dims if projection_dims != "" else None
        self.activation = nn.PReLU
        self.Loss = Loss(temperature=self.tau)

        self.project = None
        if self.project_hidden is not None:
            self.project = nn.ModuleList()
            self.activations = nn.ModuleList()
            self.project.extend(
                [nn.Linear(self.in_channels, self.project_hidden[0])])
            self.activations.extend([nn.PReLU(projection_dims[0])])
            for i in range(1, len(self.project_hidden)):
                self.project.extend(
                    [nn.Linear(self.project_hidden[i - 1], self.project_hidden[i])])
                self.activations.extend([nn.PReLU(projection_dims[i])])
            self.project.to(self.device)

        self.loss_curve = []

    def reset_parameters(self):
        pass

    def forward(self, data) -> Any:
        x = data.x.to(self.device)
        adjs = data.adjs.to(self.device)
        x = self.encoder(x, adjs)
        if self.project is not None:
            for i in range(len(self.project_hidden)):
                x = self.project[i](x)
                x = self.activations[i](x)
        return x

    def loss(self, *args, **kwargs) -> Tensor:
        pass

    def train_model(self, data: Data, cfg: CN = None, flag: str = "TRAIN MAGI") -> List:
        if cfg is None:
            cfg = self.cfg.train
        edge_index, adj = data.edge_index, data.adj

        size = self.cfg.dataset.size

        train_loader = NeighborSampler(edge_index, adj,
                                       is_train=True,
                                       node_idx=None,
                                       wt=self.cfg.dataset.wt,
                                       wl=self.cfg.dataset.wl,
                                       sizes=size,
                                       batch_size=cfg.batchsize,
                                       shuffle=True,
                                       drop_last=True,
                                       num_workers=6)

        optimizer = torch.optim.Adam(self.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
        # train
        for epoch in range(1, cfg.max_epoch + 1):
            self.train()
            total_loss = total_examples = 0

            for (batch_size, n_id, adjs), adj_batch, batch in train_loader:
                # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
                if len(self.cfg.model.dims.encoder[1:]) == 1:
                    adjs = [adjs]
                adjs = [adj.to(self.device) for adj in adjs]
                data.adjs = adjs
                adj_ = get_mask(adj_batch)
                optimizer.zero_grad()
                out = self.forward(data)
                out = F.normalize(out, p=2, dim=1)
                loss = self.Loss(out, adj_)

                loss.backward()
                optimizer.step()
                total_loss += float(loss)
                total_examples += batch_size

            self.loss_curve.append(total_loss / total_examples)
            self.logger.loss(epoch, total_loss / total_examples)
            if self.cfg.evaluate.each:
                self.evaluate(data)
        return self.loss_curve

    def get_embedding(self, data) -> Tensor:
        # eval
        edge_index, adj = data.edge_index, data.adj
        size = self.cfg.dataset.size

        test_loader = NeighborSampler(edge_index, adj,
                                      is_train=False,
                                      node_idx=None,
                                      sizes=size,
                                      batch_size=10000,
                                      shuffle=False,
                                      drop_last=False,
                                      num_workers=6)
        with torch.no_grad():
            self.eval()
            z = []
            for count, ((batch_size, n_id, adjs), _, batch) in enumerate(test_loader):
                if len(self.cfg.model.dims.encoder[1:]) == 1:
                    adjs = [adjs]
                adjs = [adj.to(self.device) for adj in adjs]
                data.adjs = adjs
                out = self.forward(data)
                z.append(out.detach().cpu().float())
            embedding = torch.cat(z, dim=0)
            embedding = F.normalize(embedding, p=2, dim=1)
        return embedding

    def clustering(self, data) -> Tuple[Tensor, Tensor, Any]:
        embedding = self.get_embedding(data)
        labels, clustering_centers = clustering(embedding.cpu().numpy(), self.cfg.dataset.n_clusters, data.y,
                                                kmeans_device=self.cfg.train.kmeans.device,
                                                batch_size=self.cfg.train.kmeans.batch, tol=1e-4, device=self.cfg.device,
                                                spectral_clustering=False)
        return embedding, labels, clustering_centers

    def evaluate(self, data):
        embedding, labels, clustering_centers = self.clustering(data)
        ground_truth = data.y.numpy()
        metric = DGCMetric(ground_truth, labels.numpy(), embedding, data.edge_index)
        metric.evaluate_one_epoch(self.logger, acc=True, nmi=True, ari=True, f1=True, hom=True, com=True, pur=True,
                                  sc=True, gre=True)
