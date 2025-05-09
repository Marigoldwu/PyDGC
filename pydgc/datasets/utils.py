# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import os.path as osp

from torch import Tensor
from torch.utils.data import Dataset as TorchDataset
from ogb.nodeproppred import PygNodePropPredDataset
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data, InMemoryDataset, Dataset, download_google_url
from torch_geometric.graphgym.loader import set_dataset_attr
from torch_geometric.utils import index_to_mask, to_undirected
from torch_geometric.datasets import (Planetoid, Coauthor, Amazon, WebKB, Actor, CitationFull,
                                      CoraFull, AttributedGraphDataset, NELL, Reddit, Reddit2, Yelp, AmazonProducts,
                                      LastFMAsia, Airports, HeterophilousGraphDataset)

PYG_SUPPORTED_DATASET = ["CORA", "CITE", "CITESEER", "PUBMED", "COCS", "COPS", "AMAC", "AMAP", "CORNELL",
                         "TEXAS", "WISC", "ACTOR", "DBLPFULL", "CORAFULL", "WIKI", "BLOG", "PPI",
                         "FLICKR", "FACEBOOK", "TWEIBO", "MAG", "NELL", "REDDIT", "REDDIT2",
                         "YELP", "AMP", "LFMA", "BAT", "EAT", "ROMAN"]
DGC_SUPPORTED_DATASET = ["ACM", "DBLP", "UAT"]
NONGRAPH_SUPPORTED_DATASET = ["USPS", "HHAR", "REUT"]
OGB_SUPPORTED_DATASET = ["ARXIV"]

DATASET_NAME_MAP = {
    "ACM": 'ACM',
    "DBLP": 'DBLP',
    "UAT": 'UAT',
    "BAT": 'Brazil',
    "EAT": 'Europe',
    "USPS": 'USPS',
    "HHAR": 'HHAR',
    "REUT": 'REUT',
    "CORA": 'Cora',
    "CITESEER": 'CiteSeer',
    "CITE": 'CiteSeer',
    "PUBMED": 'PubMed',
    "COCS": 'CS',
    "COPS": 'Physics',
    "AMAC": 'Computers',
    "AMAP": 'Photo',
    "CORNELL": 'Cornell',
    "TEXAS": 'Texas',
    "WISC": 'Wisconsin',
    "ACTOR": 'Actor',
    "DBLPFULL": 'DBLP',
    "CORAFULL": 'Corafull',
    "WIKI": 'Wiki',
    "BLOG": 'BlogCatalog',
    "PPI": 'PPI',
    "FLICKR": 'Flickr',
    "FACEBOOK": 'Facebook',
    "TWEIBO": 'TWeibo',
    "MAG": 'MAG',
    "NELL": 'NELL',
    "REDDIT": 'Reddit',
    "REDDIT2": 'Reddit',
    "YELP": 'Yelp',
    "AMP": 'AmazonProducts',
    "LFMA": 'LastFMAsia',
    "ROMAN": 'roman-empire',
    "ARXIV": "ogbn-arxiv"
}
METRIC_MAP = {"USPS": "heat", "HHAR": "cosine", "REUT": "cosine"}


class UserDataset(InMemoryDataset):
    def __init__(self, root: str, dataset_name: str):
        """
        User custom Dataset
        :param root: Path of data stored
        :param dataset_name: Name of dataset
        """
        self.dataset_name = dataset_name.upper()
        super().__init__(root)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> str:
        return f"{self.dataset_name.upper()}.npz"

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def download(self):
        pass

    def process(self):
        data_path = os.path.join(self.root, 'raw', self.raw_file_names)
        raw_data = np.load(data_path, allow_pickle=True)
        x = torch.from_numpy(raw_data['feature']).to(torch.float32)
        graph = torch.from_numpy(raw_data['graph'])
        edge_index = graph.nonzero().t().to(torch.int64)
        y = torch.from_numpy(raw_data['label']).to(torch.int64)
        data = Data(x=x, edge_index=edge_index, y=y)
        self.save([data], self.processed_paths[0])


class NonGraphDataset(InMemoryDataset):
    def __init__(self,
                 root: str,
                 dataset_name: str,
                 neighbors: int = 1,
                 metric: str = 'minkowski',
                 p: int = 2):
        """
        Dataset object for constructing non-graph data
        :param root: Path of data stored
        :param dataset_name: Name of dataset
        :param neighbors: k for knn
        :param metric: Similarity measurement
        :param p: Power parameter for the Minkowski metric.
        """
        self.root = root
        self.dataset_name = dataset_name.split('_')[0].upper()
        self.neighbors = neighbors
        self.metric = metric
        self.p = p
        super().__init__(root)
        self.load(self.processed_paths[0])

    def download(self):
        pass

    @property
    def raw_file_names(self) -> str:
        return f"{self.dataset_name.upper()}.npz"

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, f"{self.neighbors}NN", 'processed')

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def process(self):
        data_path = os.path.join(self.root, 'raw', self.raw_file_names)
        raw_data = np.load(data_path, allow_pickle=True)
        x = raw_data['feature']
        if self.metric == 'heat':
            graph_tensor = heat_kernel_knn_graph(x, self.neighbors)
        else:
            graph = kneighbors_graph(x, n_neighbors=self.neighbors, mode='connectivity', metric=self.metric, p=self.p, include_self=False, n_jobs=-1)
            graph_tensor = torch.from_numpy(graph.toarray())
        edge_index = graph_tensor.nonzero().t()
        x_tensor = torch.from_numpy(x)
        y = torch.from_numpy(raw_data['label'])
        data = Data(x=x_tensor, edge_index=edge_index, y=y)
        self.save([data], self.processed_paths[0])


def heat_kernel_knn_graph(x: np.array, k: int) -> Tensor:
    xy = np.matmul(x, x.transpose())
    xx = (x * x).sum(1).reshape(-1, 1)
    xx_yy = xx + xx.transpose()
    euclidean_distance = xx_yy - 2 * xy
    euclidean_distance[euclidean_distance < 1e-5] = 0
    distance_matrix = np.sqrt(euclidean_distance)
    # heat kernel, exp^{- euclidean^2/t}
    distance_matrix = - (distance_matrix ** 2) / 2
    distance_matrix = np.exp(distance_matrix)
    # top k
    distance_matrix = torch.from_numpy(distance_matrix)
    top_k, index = torch.topk(distance_matrix, k)
    top_k_min = torch.min(top_k, dim=-1).values.unsqueeze(-1).repeat(1, distance_matrix.shape[-1])
    ones = torch.ones_like(distance_matrix)
    zeros = torch.zeros_like(distance_matrix)
    knn_graph = torch.where(torch.ge(distance_matrix, top_k_min), ones, zeros)
    return knn_graph


class DGCGraphDataset(UserDataset):
    def __init__(self, root, dataset_name):
        super().__init__(root, dataset_name)

    def download(self) -> None:
        file_id = '1QQK4-5hMcP5MitE3vu6hnBydiunQEaPh' if self.dataset_name == 'ACM' else '1n614RUq-SLh_b3xxffaP5bgt1lBxM_OJ'
        folder = osp.join(self.root, 'raw')
        filename = f'{self.dataset_name}.npz'
        download_google_url(file_id, folder, filename)


class DGCNonGraphDataset(NonGraphDataset):
    def __init__(self, root, dataset_name, neighbors=1, metric='minkowski', p=2):
        super().__init__(root, dataset_name, neighbors, metric, p)

    def download(self) -> None:
        file_id_dict = {
            'USPS': '1d-PBz2Hk3ZHbgr4QeaZuD7Dsk1Qw21jw',
            'HHAR': '1bCBvv3ENYScXPf0uST9tZ1diaSnK5aLg',
            'REUT': '1b4MV5a-B3kHqDFlj59lgpzDhDjawB3gA'
        }
        file_id = file_id_dict[self.dataset_name]
        folder = osp.join(self.root, 'raw')
        filename = f'{self.dataset_name}.npz'
        download_google_url(file_id, folder, filename)


def load_pyg(dataset_dir: str, dataset_name: str) -> Dataset:
    if dataset_name in ["CORA", "CITE", "CITESEER", "PUBMED"]:
        return Planetoid(dataset_dir, name=DATASET_NAME_MAP[dataset_name])
    if dataset_name in ["BAT", "EAT", "UAT"]:
        return Airports(dataset_dir, name=DATASET_NAME_MAP[dataset_name])
    if dataset_name in ["COCS", "COPS"]:
        return Coauthor(dataset_dir, name=DATASET_NAME_MAP[dataset_name])
    if dataset_name in ["AMAC", "AMAP"]:
        return Amazon(dataset_dir, name=DATASET_NAME_MAP[dataset_name])
    if dataset_name in ["CORNELL", "TEXAS", "WISC"]:
        return WebKB(dataset_dir, name=DATASET_NAME_MAP[dataset_name])
    if dataset_name in ["WIKI", "BLOG", "PPI", "FLICKR", "FACEBOOK", "TWEIBO", "MAG"]:
        return AttributedGraphDataset(dataset_dir, name=DATASET_NAME_MAP[dataset_name])
    if dataset_name == "ACTOR":
        return Actor(dataset_dir)
    if dataset_name == "CORAFULL":
        return CoraFull(dataset_dir)
    if dataset_name == "DBLPFULL":
        return CitationFull(dataset_dir, name=DATASET_NAME_MAP[dataset_name])
    if dataset_name == "NELL":
        return NELL(dataset_dir)
    if dataset_name == "REDDIT":
        return Reddit(dataset_dir)
    if dataset_name == "REDDIT2":
        return Reddit2(dataset_dir)
    if dataset_name == "YELP":
        return Yelp(dataset_dir)
    if dataset_name == "AMP":
        return AmazonProducts(dataset_dir)
    if dataset_name == "LFMA":
        return LastFMAsia(dataset_dir)
    if dataset_name == "ROMAN":
        return HeterophilousGraphDataset(dataset_dir, name=DATASET_NAME_MAP[dataset_name])


def load_dgc_graph(dataset_dir: str, dataset_name: str) -> Dataset:
    return DGCGraphDataset(dataset_dir, dataset_name)


def load_dgc_non_graph(dataset_dir: str,
                       dataset_name: str,
                       *,
                       neighbors: int = 1,
                       metric: str = 'minkowski',
                       p: int = 2) -> Dataset:
    """
    Load non-graph dataset.
    :param dataset_dir: Dataset stored root path.
    :param dataset_name: Dataset name for non-graph dataset.
    :param neighbors: K for KNN. Self is not included.
    :param metric: Distance type, 'minkowski' for default.
    :param p: Power parameter for the Minkowski metric. When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
    :return: NonGraphDataset
    """
    return DGCNonGraphDataset(dataset_dir, dataset_name, neighbors, metric, p)


def load_ogb(dataset_dir: str, dataset_name: str) -> Dataset:
    dataset = PygNodePropPredDataset(root=dataset_dir, name=dataset_name)
    splits = dataset.get_idx_split()
    split_names = ['train_mask', 'val_mask', 'test_mask']
    for i, key in enumerate(splits.keys()):
        mask = index_to_mask(splits[key], size=dataset.data.y.shape[0])
        set_dataset_attr(dataset, split_names[i], mask, len(mask))
    edge_index = to_undirected(dataset.data.edge_index)
    set_dataset_attr(dataset, 'edge_index', edge_index,
                     edge_index.shape[1])
    return dataset


def load_dataset(dataset_dir: str, dataset_name: str, p: int = 2) -> Dataset:
    """
    load raw datasets.

    :param dataset_dir:
    :param dataset_name:
    :param p:
    :return: a list of networkx/deepsnap graphs, plus additional info if needed
    """
    try:
        dataset_dir = dataset_dir.split('_')[0] if dataset_dir.__contains__('_') else dataset_dir
        neighbors = int(dataset_name.split('_')[-1]) if dataset_name.__contains__('_') else 1
        dataset_name = dataset_name.split('_')[0] if dataset_name.__contains__('_') else dataset_name
        if dataset_name in OGB_SUPPORTED_DATASET:
            return load_ogb(dataset_dir, DATASET_NAME_MAP[dataset_name])
        elif dataset_name in PYG_SUPPORTED_DATASET:
            return load_pyg(dataset_dir, dataset_name)
        elif dataset_name in DGC_SUPPORTED_DATASET:
            return load_dgc_graph(dataset_dir, dataset_name)
        elif dataset_name in NONGRAPH_SUPPORTED_DATASET:
            return load_dgc_non_graph(dataset_dir, dataset_name, neighbors=neighbors, metric=METRIC_MAP[dataset_name[:4]], p=p)
        else:
            raise ValueError
    except NotADirectoryError:
        print(f"{dataset_dir} is not a directory!")
    except ValueError:
        print(f"Dataset name {dataset_name} is unsupported! Must be selected from {str(PYG_SUPPORTED_DATASET + DGC_SUPPORTED_DATASET + NONGRAPH_SUPPORTED_DATASET + OGB_SUPPORTED_DATASET)}")
    except Exception as e:
        print(f"Unknown error occurred: {e}")


def preprocess_custom_data(root: str, dataset_name: str, dataset_type: str = 'graph'):
    """
    Transform dataset with format from Awesome-Deep-Graph-Clustering.
    :param root: root path
    :param dataset_name: Dataset name.
    :param dataset_type: Dataset type. Options: 'graph', 'non-graph'
    :return:
    """
    try:
        if not osp.isdir(root):
            raise NotADirectoryError(f"{root} is not a directory!")
        if dataset_type not in ['graph', 'non-graph']:
            raise ValueError(f"Dataset type {dataset_type} is unsupported! Supported: 'graph', 'non-graph'.")
        save_path = osp.join(root, f'{dataset_name.upper()}/raw/{dataset_name.upper()}.npz')
        f_path = osp.join(root, f"{dataset_name}/raw/feature.npy")
        if not osp.exists(f_path):
            raise FileNotFoundError(f"{f_path} not found!")
        feature = np.load(f_path, allow_pickle=True)

        l_path = osp.join(root, f"{dataset_name}/raw/label.npy")
        if not osp.exists(l_path):
            raise FileNotFoundError(f"{l_path} not found!")
        label = np.load(l_path, allow_pickle=True)

        if dataset_type == 'graph':
            g_path = osp.join(root, f"{dataset_name}/raw/graph.npy")
            if not osp.exists(g_path):
                raise FileNotFoundError(f"{g_path} not found!")
            graph = np.load(g_path, allow_pickle=True)
            np.savez(save_path, feature=feature, graph=graph, label=label)
        else:
            np.savez(save_path, feature=feature, label=label)
    except NotADirectoryError as e:
        print(e)
    except FileNotFoundError as e:
        print(e)
    except ValueError as e:
        print(e)
    except Exception as e:
        print(e)
    return None


class LoadAttribute(TorchDataset):
    def __init__(self, x):
        if isinstance(x, torch.Tensor) and x.device != torch.device('cpu'):
            x = x.cpu()
        self.x = x

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))
