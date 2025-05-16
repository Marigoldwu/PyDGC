# -*- coding: utf-8 -*-
import os
import umap
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Tuple

from matplotlib.ticker import FixedLocator, FixedFormatter
from sklearn.manifold import TSNE
from sklearn.metrics import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

from . import get_formatted_time


class DGCVisual:
    def __init__(self,
                 save_path: str = '.',
                 save_format: str = 'png',
                 font_family: str or list = 'sans-serif',
                 font_size: int = 20):
        time_ = get_formatted_time()
        self.save_path = os.path.join(save_path, time_)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
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
                        show_axis: bool = False,
                        legend: bool = False,
                        dpi: int = 300,
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
        :param dpi:
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
        sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=labels, palette=palette, legend=legend)
        file_path = f"{self.save_path}/{filename}.{self.save_format}"
        plt.savefig(file_path, dpi=dpi, bbox_inches='tight')
        plt.clf()

    def plot_heatmap(self,
                     data: np.array,
                     labels: np.array,
                     method: str = 'inner_product',
                     color_map="YlGnBu",
                     fig_size: Tuple[int, int] = (8, 8),
                     filename: str = "heatmap_plot",
                     show_color_bar: bool = False,
                     show_axis: bool = False,
                     dpi: int = 300):
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
        :param dpi:
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
        plt.tight_layout()
        plt.savefig(file_path, dpi=dpi, bbox_inches='tight')
        plt.clf()

    def plot_loss(self,
                  losses: list,
                  metrics: list = None,
                  metrics_name: str = None,
                  fig_size: Tuple[int, int] = (8/2.54, 6/2.54),
                  marker: str = 'o',
                  line_style: str = '-',
                  color: str = 'blue',
                  line_width: int = 2,
                  title: str = None,
                  dpi: int = 300,
                  filename: str = "loss_curve_plot"):
        """
        绘制损失曲线

        :param losses: 损失值列表
        :param metrics:
        :param metrics_name:
        :param fig_size: 图片尺寸
        :param losses:
        :param fig_size:
        :param marker:
        :param line_style:
        :param color:
        :param line_width:
        :param title: 图形的标题
        :param dpi:
        :param filename: 保存图像的文件名
        :return:
        """
        epochs = np.arange(1, len(losses) + 1)
        losses = np.array(losses)
        color = (0.4, 0.4, 0.8)
        acc_color = (0.9, 0.4, 0.0)
        if metrics is None:
            plt.figure(figsize=fig_size, dpi=dpi)

            plt.plot(epochs, losses, marker=marker, linestyle=line_style, color=color, linewidth=line_width)
            # plt.xlabel('Epochs')
            # plt.ylabel('Loss')
            if title is not None:
                plt.title(title)

        else:
            metrics = np.array(metrics)
            # 创建图像和双Y轴
            fig, ax1 = plt.subplots(figsize=fig_size, dpi=dpi)

            # 设置左侧Y轴 (损失函数)
            color1 = color
            color2 = acc_color
            # ax1.set_xlabel('Epochs')
            # ax1.set_ylabel('Loss', color=color1)
            ax1.plot(epochs, losses, linestyle=line_style, color=color1, linewidth=line_width)
            ax1.tick_params(axis='y', labelcolor=color1)
            ax1.tick_params(axis='x')

            # 设置右侧Y轴 (准确率)
            ax2 = ax1.twinx()
            # color2 = 'tab:blue'
            # ax2.set_ylabel(f'{metrics_name}')
            ax2.plot(epochs, metrics, linestyle='--', color=color2, linewidth=line_width)
            ax2.tick_params(axis='y', labelcolor=color2)

            # loss_min = np.min(losses)
            # loss_max = np.max(losses)
            # ax1.yaxis.set_major_locator(FixedLocator([loss_min, loss_max]))
            # ax1.yaxis.set_major_formatter(FixedFormatter([f'{loss_min:.2f}', f'{loss_max:.2f}']))
            #
            # # 对于右侧Y轴（准确率）
            # acc_min = np.min(metrics)
            # acc_max = np.max(metrics)
            # ax2.yaxis.set_major_locator(FixedLocator([acc_min, acc_max]))
            # ax2.yaxis.set_major_formatter(FixedFormatter([f'{acc_min:.3f}', f'{acc_max:.3f}']))

            # 设置X轴仅显示最小值和最大值
            epoch_min = np.min(epochs)
            epoch_max = np.max(epochs)
            ax1.xaxis.set_major_locator(FixedLocator([epoch_min, epoch_max]))
            ax1.xaxis.set_major_formatter(FixedFormatter([f'{epoch_min}', f'{epoch_max}']))
            ax1.set_yticks([])  # 隐藏左侧Y轴刻度
            ax2.set_yticks([])  # 隐藏右侧Y轴刻度
        # 添加标题
        if title is not None:
            plt.title(title)

        # 调整布局
        plt.tight_layout()
        file_path = f"{self.save_path}/{filename}.{self.save_format}"
        plt.savefig(file_path, dpi=dpi, bbox_inches='tight')
        plt.clf()
