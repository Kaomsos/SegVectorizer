from __future__ import annotations
from typing import TYPE_CHECKING, List
if TYPE_CHECKING:
    from ..typing_ import SingleConnectedComponent, WallCenterLine

import importlib
from abc import ABC, abstractmethod

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.optim import RMSprop

from ..optimize import objective
from entity.wall_center_line import WallCenterLineWithOpenPoints as WCL_O
from ..entity.polygon import Rectangle
from ..geometry import distance_seg_to_segments, project_seg_to_seg
from ..optimize.objective import log_iou
from ..optimize.softras import FixedCenterRectangle2DRasterizer as RectangleRasterizer


###########################################
# fitting rectangles
class RectangleFitter(ABC):
    @abstractmethod
    def fit(self, target: SingleConnectedComponent) -> Rectangle:
        pass


class PCAFitter(RectangleFitter):
    def __init__(self):
        self.pca = PCA(n_components=2)

    def fit(self, target: SingleConnectedComponent) -> Rectangle:
        X = np.argwhere(target.array)[..., [1, 0]]
        assert len(X.shape) == 2
        assert X.shape[1] == 2

        X_trans = self.pca.fit_transform(X)

        center = X.mean(axis=0)
        w_vec, h_vec = self.pca.components_
        w, h = X_trans.max(axis=0) - X_trans.min(axis=0)

        rect = Rectangle(w=w, h=h, center=center, w_vec=w_vec, h_vec=h_vec)
        return rect


class SoftRasFitter(RectangleFitter):
    def __init__(self, sigma=0.1, max_iter=5, lr=0.1, log_iou_target=-0.5):
        self._rasterizer = None
        self._optimizer = None

        self._sigma = sigma

        self._max_iter = max_iter
        self._log_iou_target = log_iou_target
        self._i = None
        self._lr = lr
        self._verbose = None

        self._w, self._h, self._theta = None, None, None
        self._loss = None

        self._cc = None
        self._rendered = None

    def fit(self, target: SingleConnectedComponent) -> Rectangle:
        self._fit(target, verbose=False)
        center, (w, h, theta) = self.result
        rect = Rectangle(center=center, w=w, h=h, theta=theta)
        return rect

    def _fit(self,
             component: SingleConnectedComponent,
             verbose=False,
             ) -> None:
        self._cc = component
        self._verbose = verbose
        self._rasterizer = RectangleRasterizer(self._cc.size,
                                               center=self._cc.center,
                                               sigma=self._sigma,
                                               mode="soft euclidean")

        self._w, self._h = component.width_height
        self._w = torch.tensor(self._w, dtype=torch.float64, requires_grad=True)
        self._h = torch.tensor(self._h, dtype=torch.float64, requires_grad=True)
        self._theta = torch.tensor(0, dtype=torch.float64, requires_grad=True)

        self._optimizer = RMSprop([self._w, self._h, self._theta], lr=self._lr)

        for i in range(self._max_iter):
            self._i = i
            self._rendered = self._rasterizer.rasterize(self._w, self._h, self._theta)
            self._loss = -log_iou(rendered=self._rendered, target=component.array)
            if -self._loss >= self._log_iou_target:
                break
            self._loss.backward()
            self._optimizer.step()
            self._hook_after_step()

    def _hook_after_step(self):
        if self._verbose:
            self.plot_soft_ras()
            print((f'iter = {self._i}, \n'
                   f'-log_iou = {self._loss.detach().numpy()}, \n'
                   f'grad_w = {self._w.grad.detach().numpy()}, \n'
                   f'grad_h = {self._h.grad.detach().numpy()}, \n'
                   f'grad_theta = {self._theta.grad.detach().numpy()}, \n'))

    def plot_soft_ras(self):
        plt = importlib.import_module('.pyplot', 'matplotlib')
        rendered = self._rendered
        rect_img = self._cc
        plt.subplot(1, 2, 1)
        plt.imshow(rendered.detach().numpy())
        plt.title(f"sigma = {self._sigma}, iter = {self._i}")
        plt.subplot(1, 2, 2)
        plt.imshow(rect_img.array + rendered.detach().numpy())
        plt.title(f"IOU={np.exp(-self._loss.detach().numpy()): .5f}")
        plt.show()

    @property
    def w(self):
        return self._w.detach().numpy()

    @property
    def h(self):
        return self._h.detach().numpy()

    @property
    def theta(self):
        return self._theta.detach().numpy()

    @property
    def result(self):
        return self._cc.objective.center, \
               (float(self.w), float(self.h), float(self.theta))


def fit_open_points(target: SingleConnectedComponent, fitter=PCAFitter()) -> Rectangle:
    """
    a wrapper for Rectangle Fitter
    :param target:
    :param fitter:
    :return:
    """
    rect = fitter.fit(target)
    return rect


##########################################################
# compute the final position for open points
def insert_open_points_in_wcl(opens: List[Rectangle], wcl: WallCenterLine) -> WCL_O:
    wcl_o = WCL_O.from_wcl(wcl)
    for rect in opens:
        segs = wcl_o.segments_collection
        seg = rect.ends
        ds = distance_seg_to_segments(seg=seg, segments=segs)

        which = ds.argmin()
        to = (segs[0][which], segs[1][which])
        e = wcl_o.i2e[which]

        proj = project_seg_to_seg(seg, to)

        if rect.tag == "window":
            wcl_o.insert_window_to_edge(proj, e)
        elif rect.tag == "door":
            wcl_o.insert_door_to_edge(proj, e)
        else:
            wcl_o.insert_open_to_edge(proj, e)

    return wcl_o
