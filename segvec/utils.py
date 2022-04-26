from __future__ import annotations
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from typing_ import WallCenterLine, Polygon, Rectangle, WallCenterLineWithOpenPoints, Contour
import warnings
import numpy as np
import torch
from matplotlib import pyplot as plt
from torchviz import make_dot

from segvec.entity.wall_center_line import WallCenterLine
from segvec.entity.image import BinaryImage
from segvec.entity.polygon import Polygon
from segvec.geometry import get_bounding_box

__all__ = ["plot_binary_image",
           "plot_labeled_image",
           "plot_empty_image",
           "plot_empty_image_like",
           "plot_contours",
           'plot_polygon_comparing_cc',
           'plot_polygon',
           'plot_room_contours',
           'plot_rooms_in_wcl',
           'plot_wall_center_lines',
           'plot_wcl_against_target',
           'viz_computation_graph',
           'plot_position_of_rects',
           'plot_wcl_o_against_target',
           'palette',
           'plt',
           ]


def plot_binary_image(bin_arr, title="", show=True):
    plt.imshow(bin_arr, cmap='gray')
    plt.title(title)
    if show:
        plt.show()


def plot_labeled_image(bin_arr):
    plt.imshow(bin_arr, cmap='nipy_spectral')
    plt.show()


def plot_empty_image(arr_shape, show=False):
    plt.imshow(np.empty(shape=arr_shape) + np.nan)
    if show:
        plt.show()


def plot_empty_image_like(arr, show=False):
    plt.imshow(np.empty(shape=arr.shape) + np.nan)
    if show:
        plt.show()


def plot_contours(connected_component, show=True):
    warnings.filterwarnings('ignore')
    plt.imshow(connected_component.array + np.nan, cmap='gray')
    for c in connected_component.contours:
        plt.plot(c[..., 0], c[..., 1])
    if show:
        plt.show()
    warnings.filterwarnings('default')


def plot_polygon_comparing_cc(bin_img: BinaryImage, plg: Polygon):
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


def plot_polygon(plg: Polygon, show=False):
    plt.plot(plg.plot_x, plg.plot_y, marker='+')
    if show:
        plt.show()


def plot_room_contours(contours: List[Contour], show=False):
    for c in contours:
        plot_polygon(c, show=False)
    if show:
        plt.show()


def plot_wall_center_lines(wcl: WallCenterLine,
                           widths=False,
                           annotation=True,
                           title: str = "",
                           show=True):
    sps, eps = wcl.segments_collection
    wcl_widths = wcl.widths
    for i, (sp, ep) in enumerate(zip(sps, eps)):
        xs = [p[0] for p in [sp, ep]]
        ys = [p[1] for p in [sp, ep]]
        if widths:
            plt.plot(xs, ys, color='#3399ff', lw=wcl_widths[i] * 72 / plt.gcf().dpi)
        else:
            plt.plot(xs, ys, color='#3399ff')

    if annotation:
        pos = wcl.V
        i2v = wcl.i2v
        for i, p in enumerate(pos):
            plt.text(p[0], p[1], str(i2v[i]), size='x-small')

    plt.title(title)
    if show:
        plt.show()


def plot_wcl_against_target(wcl,
                            target=None,
                            title='',
                            widths=False,
                            annotation=True,
                            show=True):
    if target is not None:
        plt.imshow(1 - target, cmap='gray')
    plot_wall_center_lines(wcl, title=title, widths=widths, annotation=annotation, show=False)
    if show:
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


def plot_position_of_rects(l: List[Rectangle], color="#3399ff", show=False):
    for i, rect in enumerate(l):
        center = rect.center
        ends = rect.ends
        plt.plot(ends[..., 0], ends[..., 1], color=color, lw=min(rect.w, rect.h) * 72 / plt.gcf().dpi)
        plt.text(center[0] + 5, center[1] - 5, f"{i}", size='x-small')

    if show:
        plt.show()


def plot_wcl_o_against_target(wcl_o: WallCenterLineWithOpenPoints,
                              target: np.ndarray = None,
                              title='',
                              widths=False,
                              annotation=True,
                              show=True):
    plot_wcl_against_target(wcl_o, target,
                            title=title,
                            widths=widths,
                            annotation=annotation,
                            show=False)

    if widths:
        windows_widths = wcl_o.windows_widths
        doors_widths = wcl_o.doors_widths

    for i, e in enumerate(wcl_o.windows):
        p1, p2 = e
        if widths:
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], lw=windows_widths[i] * 72 / plt.gcf().dpi, color='red')
        else:
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='red')

    for i, e in enumerate(wcl_o.doors):
        p1, p2 = e
        if widths:
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], lw=doors_widths[i] * 72 / plt.gcf().dpi, color='lime')
        else:
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='lime')

    if show:
        plt.show()


def plot_rooms_in_wcl(wcl: WallCenterLine,
                      palette: dict = None,
                      title: str = "",
                      contour: bool = True,
                      show: bool = False
                      ):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    reverse_palette = {v: k for k, v in palette.items()}

    for room, type_ in zip(wcl.rooms, wcl.room_types):
        indices = list(range(room.shape[0])) + [0]
        p1, p2 = get_bounding_box(room)
        x, y = (p1 + p2) / 2

        if palette is None:
            text = f'type {type_}'
        else:
            text = reverse_palette[type_]

        if contour:
            plt.plot(room[indices, 0], room[indices, 1])

        plt.text(x, y, text, size='x-small', ha='center', va='center')

    plt.title(title)

    if show:
        plt.show()


palette = {
     'background': (255, 255, 255),
     'closet': (192, 192, 224),
     'bathroom/washroom': (192, 255, 255),
     'livingroom/kitchen/dining add_room': (224, 255, 192),
     'bedroom': (255, 224, 128),
     'hall': (255, 160, 96),
     'balcony': (255, 224, 224),
     7: (224, 224, 224),  # not used
     8: (224, 224, 128),  # not used
     'door&window': (255, 60, 128),
     'wall': (0, 0, 0)
}