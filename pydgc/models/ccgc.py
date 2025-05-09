# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import Tuple, List, Any

from torch import Tensor
from torch_geometric.data import Data

from . import DGCModel
from ..clusterings import KMeansGPU
from ..metrics import DGCMetric
from ..utils import Logger

from yacs.config import CfgNode as CN


def init_clustering(feature, cluster_num):
    kmeans = KMeansGPU(n_clusters=cluster_num, distance="euclidean", device="cuda")
    predict_labels, _ = kmeans.fit(feature)
    dis = kmeans.pairwise_distance(feature, kmeans.cluster_centers_)
    return predict_labels, dis


class CCGC(DGCModel):
    def __init__(self, logger: Logger, cfg: CN):
        super(CCGC, self).__init__(logger, cfg)
        self.device = torch.device(cfg.device)
        dims = cfg.model.dims.copy()
        dims.insert(0, cfg.dataset.num_features)
        self.layers1 = nn.Linear(dims[0], dims[1]).to(self.device)
        self.layers2 = nn.Linear(dims[0], dims[1]).to(self.device)

        self.loss_curve = []
        self.results = {}

    def reset_parameters(self):
        pass

    def forward(self, x) -> Any:
        x = x.to(self.device)
        out1 = self.layers1(x)
        out2 = self.layers2(x)

        out1 = F.normalize(out1, dim=1, p=2)
        out2 = F.normalize(out2, dim=1, p=2)

        return out1, out2

    def loss(self, *args, **kwargs) -> Tensor:
        pass

    def train_model(self, data: Data, cfg: CN = None, flag: str = "TRAIN CCGC") -> List:
        if cfg is None:
            cfg = self.cfg.train
        smooth_fea = data.x.to(self.device)
        predict_labels, dis = init_clustering(smooth_fea, self.cfg.dataset.n_clusters)

        optimizer = torch.optim.Adam(self.parameters(), lr=float(cfg.lr))

        sample_size = self.cfg.dataset.num_nodes
        target = torch.eye(sample_size).to(self.device)

        for epoch in range(cfg.max_epoch):
            self.train()
            z1, z2 = self.forward(smooth_fea)
            if epoch > 50:

                high_confidence = torch.min(dis.cpu(), dim=1).values
                threshold = torch.sort(high_confidence).values[int(len(high_confidence) * cfg.threshold)]
                high_confidence_idx = np.argwhere(high_confidence < threshold)[0]

                # pos samples
                index = torch.tensor(range(sample_size), device=self.device)[high_confidence_idx]
                y_sam = predict_labels.detach().clone()[high_confidence_idx].to(self.device)
                index = index[torch.argsort(y_sam)]
                class_num = {}

                for label in torch.sort(y_sam).values:
                    label = label.item()
                    if label in class_num.keys():
                        class_num[label] += 1
                    else:
                        class_num[label] = 1
                key = sorted(class_num.keys())
                if len(class_num) < 2:
                    continue
                pos_contrastive = 0
                centers_1 = torch.tensor([], device=self.device)
                centers_2 = torch.tensor([], device=self.device)

                for i in range(len(key[:-1])):
                    class_num[key[i + 1]] = class_num[key[i]] + class_num[key[i + 1]]
                    now = index[class_num[key[i]]:class_num[key[i + 1]]]
                    pos_embed_1 = z1[np.random.choice(now.cpu(), size=int((now.shape[0] * 0.8)), replace=False)]
                    pos_embed_2 = z2[np.random.choice(now.cpu(), size=int((now.shape[0] * 0.8)), replace=False)]
                    pos_contrastive += (2 - 2 * torch.sum(pos_embed_1 * pos_embed_2, dim=1)).sum()
                    centers_1 = torch.cat([centers_1, torch.mean(z1[now], dim=0).unsqueeze(0)], dim=0)
                    centers_2 = torch.cat([centers_2, torch.mean(z2[now], dim=0).unsqueeze(0)], dim=0)

                pos_contrastive = pos_contrastive / self.cfg.dataset.n_clusters
                if pos_contrastive == 0:
                    continue
                if len(class_num) < 2:
                    loss = pos_contrastive
                else:
                    centers_1 = F.normalize(centers_1, dim=1, p=2)
                    centers_2 = F.normalize(centers_2, dim=1, p=2)
                    S = centers_1 @ centers_2.T
                    S_diag = torch.diag_embed(torch.diag(S))
                    S = S - S_diag
                    neg_contrastive = F.mse_loss(S, torch.zeros_like(S))
                    loss = pos_contrastive + cfg.alpha * neg_contrastive
            else:
                S = z1 @ z2.T
                loss = F.mse_loss(S, target)

            loss.backward(retain_graph=True)
            optimizer.step()
            self.loss_curve.append(loss.item())
            self.logger.loss(epoch, loss)

            if epoch % 10 == 0:
                self.eval()
                z1, z2 = self.forward(smooth_fea)
                embedding = (z1 + z2) / 2
                predict_labels, dis = init_clustering(embedding, self.cfg.dataset.n_clusters)
            if self.cfg.evaluate.each:
                self.evaluate(data)
        return self.loss_curve

    def get_embedding(self, data) -> Tensor:
        x = data.x.to(self.device)
        with torch.no_grad():
            z1, z2 = self.forward(x)
            embedding = (z1 + z2) / 2
            return embedding.detach()

    def clustering(self, data) -> Tuple[Tensor, Tensor, Tensor]:
        embedding = self.get_embedding(data)
        labels_, clustering_centers_ = KMeansGPU(self.cfg.dataset.n_clusters).fit(embedding)
        return embedding, labels_, clustering_centers_

    def evaluate(self, data):
        embedding, labels, clustering_centers = self.clustering(data)
        ground_truth = data.y.numpy()
        metric = DGCMetric(ground_truth, labels.numpy(), embedding, data.edge_index)
        metric.evaluate_one_epoch(self.logger, acc=True, nmi=True, ari=True, f1=True, hom=True, com=True, pur=True,
                                  sc=True, gre=True)
