from __future__ import annotations
import numpy as np
import torch
from matplotlib import pyplot as plt
from torchviz import make_dot
from entity_class import BinaryImage, Polygon, WallCenterLine
# from typing_ import WallCenterLine, BinaryImage, Polygon
import warnings

__all__ = ["plot_binary_image",
           "plot_labeled_image",
           "plot_contours",
           'plot_polygon',
           'plot_wall_center_lines',
           'plot_wcl_against_target',
           'viz_computation_graph',]


def plot_binary_image(bin_arr, title="", show=True):
    plt.imshow(bin_arr, cmap='gray')
    plt.title(title)
    if show:
        plt.show()


def plot_labeled_image(bin_arr):
    plt.imshow(bin_arr, cmap='nipy_spectral')
    plt.show()


def plot_contours(connected_component):
    warnings.filterwarnings('ignore')
    plt.imshow(connected_component.array + np.nan, cmap='gray')
    for c in connected_component.contours:
        plt.plot(c[..., 0], c[..., 1])
    plt.show()
    warnings.filterwarnings('default')


def plot_polygon(bin_img: BinaryImage, plg: Polygon):
    warnings.filterwarnings('ignore')
    plt.subplot(1, 2, 2)
    plt.imshow(1 - bin_img.array, cmap='gray')
    plt.plot(plg.plot_x, plg.plot_y, marker='+')
    plt.title('ground truth')

    plt.subplot(1, 2, 1)
    plt.imshow(bin_img.array + np.nan, cmap='gray')
    plt.plot(plg.plot_x, plg.plot_y, marker='+')

    plt.show()
    warnings.filterwarnings('default')


def plot_wall_center_lines(wcl: WallCenterLine, annotation=True, title: str = "", show=True):
    sps, eps = wcl.segments_collection
    for sp, ep in zip(sps, eps):
        xs = [p[0] for p in [sp, ep]]
        ys = [p[1] for p in [sp, ep]]
        plt.plot(xs, ys, color='#3399ff')

    if annotation:
        pos = wcl.V
        i2v = wcl.i2v
        for i, p in enumerate(pos):
            plt.text(p[0], p[1], str(i2v[i]), size='x-small')

    plt.title(title)
    if show:
        plt.show()


def plot_wcl_against_target(target, wcl, title=''):
    plt.imshow(1 - target, cmap='gray')
    plot_wall_center_lines(wcl, title=title, show=False)
    plt.show()


def viz_computation_graph(var: torch.Tensor,
                          show_attrs=False,
                          show_saved=False,
                          to_file=True,
                          path=None
                          ) -> None:
    dot = make_dot(var, show_attrs=show_attrs, show_saved=show_saved)
    if to_file:
        assert path is not None
        dot.render(path, format='png')
    else:
        pass