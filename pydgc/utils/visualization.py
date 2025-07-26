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
    """A class for visualizing data.

    Args:
        save_path (str, optional): The path to save the images. Defaults to '.'.
        save_format (str, optional): The format of the images. Defaults to 'png'.
        font_family (str or list, optional): The font family. Defaults to 'sans-serif'.
        font_size (int, optional): The font size. Defaults to 20.
    """
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
        """Check if the save format is supported.

        Args:
            save_format (str): The save format, e.g., 'png', 'pdf', 'jpg', 'jpeg', 'bmp', 'tiff', 'gif', 'svg', 'eps'.

        Raises:
            ValueError: If the save format is not supported.
        """
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
        """Plot the clustering results with tsne or umap dimension reduction.

        Args:
            data (np.array): The input data, shape (n_samples, n_features).
            labels (np.array): The data labels.
            method (str, optional): The dimensionality reduction method, 'tsne' or 'umap'. Defaults to 'tsne'.
            palette (str, optional): The color palette. Defaults to "viridis".
            fig_size (Tuple[int, int], optional): The figure size. Defaults to (10, 8).
            filename (str, optional): The filename to save the plot. Defaults to "tsne_plot".
            show_axis (bool, optional): Whether to show the axis. Defaults to False.
            legend (bool, optional): Whether to show the legend. Defaults to False.
            dpi (int, optional): The DPI of the plot. Defaults to 300.
            random_state (int, optional): The random state. Defaults to 42.
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
        """Plot the heatmap of the data.

        Args:
            data (np.array): The input data, shape (n_samples, n_features).
            labels (np.array): The data labels.
            method (str, optional): The similarity method, 'cosine' or 'euclidean' or 'inner_product'. Defaults to 'inner_product'.
            color_map (str, optional): The color map. Defaults to "YlGnBu".
            fig_size (Tuple[int, int], optional): The figure size. Defaults to (8, 8).
            filename (str, optional): The filename to save the plot. Defaults to "heatmap_plot".
            show_color_bar (bool, optional): Whether to show the color bar. Defaults to False.
            show_axis (bool, optional): Whether to show the axis. Defaults to False.
            dpi (int, optional): The DPI of the plot. Defaults to 300.
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
        """Plot the loss curve and metrics curve if metrics valid.

        Args:
            losses (list): The loss values.
            metrics (list, optional): The metrics values. Defaults to None.
            metrics_name (str, optional): The metrics name. Defaults to None.
            fig_size (Tuple[int, int], optional): The figure size. Defaults to (8/2.54, 6/2.54).
            marker (str, optional): The marker style. Defaults to 'o'.
            line_style (str, optional): The line style. Defaults to '-'.
            color (str, optional): The line color. Defaults to 'blue'.
            line_width (int, optional): The line width. Defaults to 2.
            title (str, optional): The title. Defaults to None.
            dpi (int, optional): The DPI. Defaults to 300.
            filename (str, optional): The filename. Defaults to "loss_curve_plot".

        """
        epochs = np.arange(1, len(losses) + 1)
        losses = np.array(losses)
        color = (0.4, 0.4, 0.8)
        acc_color = (0.9, 0.4, 0.0)
        if metrics is None or len(metrics) == 0:
            plt.figure(figsize=fig_size, dpi=dpi)

            plt.plot(epochs, losses, marker=marker, linestyle=line_style, color=color, linewidth=line_width)
            # plt.xlabel('Epochs')
            # plt.ylabel('Loss')
            if title is not None:
                plt.title(title)

        else:
            metrics = np.array(metrics)
            # create the figure and double y-axis
            fig, ax1 = plt.subplots(figsize=fig_size, dpi=dpi)

            # set the left y-axis (loss)
            color1 = color
            color2 = acc_color
            ax1.plot(epochs, losses, linestyle=line_style, color=color1, linewidth=line_width)
            ax1.tick_params(axis='y', labelcolor=color1)
            ax1.tick_params(axis='x')

            # set the right y-axis (metrics)
            ax2 = ax1.twinx()
            ax2.plot(epochs, metrics, linestyle='--', color=color2, linewidth=line_width)
            ax2.tick_params(axis='y', labelcolor=color2)

            # set the x-axis to show only the minimum and maximum values
            epoch_min = np.min(epochs)
            epoch_max = np.max(epochs)
            ax1.xaxis.set_major_locator(FixedLocator([epoch_min, epoch_max]))
            ax1.xaxis.set_major_formatter(FixedFormatter([f'{epoch_min}', f'{epoch_max}']))
            ax1.set_yticks([])  # hide the left y-axis tick
            ax2.set_yticks([])  # hide the right y-axis tick
        # add title
        if title is not None:
            plt.title(title)

        # adjust the layout
        plt.tight_layout()
        file_path = f"{self.save_path}/{filename}.{self.save_format}"
        plt.savefig(file_path, dpi=dpi, bbox_inches='tight')
        plt.clf()
