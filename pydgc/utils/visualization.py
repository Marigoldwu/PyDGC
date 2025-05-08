# -*- coding: utf-8 -*-
import os
import umap
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Tuple
from sklearn.manifold import TSNE
from sklearn.metrics import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


class DGCVisual:
    def __init__(self,
                 save_path: str = '.',
                 save_format: str = 'png',
                 font_family: str or list = 'sans-serif',
                 font_size: int = 12):
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.check_save_format(save_format)
        self.save_format = save_format
        self.font_family = font_family
        self.font_size = font_size
        plt.rcParams['font.family'] = self.font_family
        plt.rcParams['font.size'] = self.font_size

    @staticmethod
    def check_save_format(save_format):
        support_format = ["png", "pdf", "jpg", "jpeg", "bmp", "tiff", "gif", "svg", "eps"]
        assert save_format in support_format

    def plot_clustering(self,
                        data: np.array,
                        labels: np.array,
                        method: str = 'tsne',
                        palette="viridis",
                        fig_size: Tuple[int, int] = (10, 8),
                        filename: str = "tsne_plot",
                        show_axis: bool = True,
                        random_state=42):
        """
        使用 t-SNE 对数据进行降维并可视化

        :param data: 输入数据，形状为 (n_samples, n_features)
        :param labels: 数据对应的标签
        :param method: 'tsne' or 'umap'
        :param palette: 颜色
        :param fig_size: 图片尺寸
        :param filename: 保存图像的文件名
        :param show_axis: 是否显示坐标轴
        :param random_state: 随机数
        """
        if method == 'tsne':
            tsne = TSNE(n_components=2, random_state=random_state)
            data = tsne.fit_transform(data)
        if method == 'umap':
            reducer = umap.UMAP(n_components=2)
            data = reducer.fit_transform(data)
            data = MinMaxScaler().fit_transform(data)
        plt.figure(figsize=fig_size)
        if not show_axis:
            plt.axis("off")
        sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=labels, palette=palette)
        file_path = f"{self.save_path}/{filename}.{self.save_format}"
        plt.savefig(file_path)
        plt.clf()

    def plot_heatmap(self,
                     data: np.array,
                     labels: np.array,
                     method: str = 'inner_product',
                     color_map="YlGnBu",
                     fig_size: Tuple[int, int] = (8, 8),
                     filename: str = "heatmap_plot",
                     show_color_bar: bool = True,
                     show_axis: bool = True):
        """
        绘制热力图

        :param data: 输入数据，二维数组
        :param labels: 用于划分簇的标签
        :param method: 相似度计算方式，'cosine' or 'euclidean' or 'inner_product'
        :param color_map: 颜色映射
        :param fig_size: 图形尺寸
        :param filename: 保存图像的文件名
        :param show_color_bar: 是否显示color bar
        :param show_axis: 是否显示坐标轴
        """
        # Sort F based on the sort indices
        sort_indices = np.argsort(labels)
        data = data[sort_indices]
        similarity = None
        if method == 'cosine':
            similarity = cosine_similarity(data)
        if method == 'euclidean':
            similarity = euclidean_distances(data)
        if method == 'inner_product':
            similarity = data @ data.T
        plt.figure(figsize=fig_size)
        plt.imshow(similarity, cmap=color_map, interpolation='nearest')
        if show_color_bar:
            plt.colorbar()
        if not show_axis:
            plt.axis("off")
        file_path = f"{self.save_path}/{filename}.{self.save_format}"
        plt.savefig(file_path)
        plt.clf()

    def plot_loss(self,
                  losses: np.array,
                  fig_size: Tuple[int, int] = (10, 8),
                  marker: str = 'o',
                  line_style: str = '-',
                  color: str = 'blue',
                  line_width: int = 2,
                  title: str = "Loss Curve",
                  filename: str = "loss_curve_plot"):
        """
        绘制损失曲线

        :param losses: 损失值列表
        :param fig_size: 图片尺寸
        :param losses:
        :param fig_size:
        :param marker:
        :param line_style:
        :param color:
        :param line_width:
        :param title: 图形的标题
        :param filename: 保存图像的文件名
        :return:
        """
        plt.figure(figsize=fig_size)
        epochs = np.arange(1, len(losses)+1)
        losses = np.array(losses)
        plt.plot(epochs, losses, marker=marker, linestyle=line_style, color=color, linewidth=line_width)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(title)
        file_path = f"{self.save_path}/{filename}.{self.save_format}"
        plt.savefig(file_path)
        plt.clf()
