from __future__ import annotations
from typing import TYPE_CHECKING, List
if TYPE_CHECKING:
    from ..typing_ import SingleConnectedComponent, WallCenterLine

import importlib
from abc import ABC, abstractmethod

import numpy as np
import torch
from sklearn.decomposition import PCA
from scipy.stats import norm
from torch.optim import RMSprop

from ..optimize import objective
from ..entity.wall_center_line import WallCenterLineWithOpenPoints as WCL_O
from ..entity.polygon import Rectangle
from ..geometry import distance_seg_to_segments, project_seg_to_seg
from ..optimize.objective import log_iou
from ..optimize.softras import FixedCenterRectangle2DRasterizer as RectangleRasterizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


###########################################
# fitting rectangles
class RectangleFitter(ABC):
    @abstractmethod
    def fit(self, target: SingleConnectedComponent) -> Rectangle:
        pass


class PCAFitter(RectangleFitter):
    __size_estimators__ = ['gaussian', 'min_max']

    def __init__(self, size_estimator: str = None):
        self.pca = PCA(n_components=2)

        if size_estimator is None:
            size_estimator = 'gaussian'
        assert size_estimator in self.__size_estimators__
        self._size_estimator = size_estimator

    def fit(self, target: SingleConnectedComponent) -> Rectangle:
        """
            plt.figure(figsize=(5, 12))
            plt.scatter(X[..., 0], X[..., 1], marker='o', facecolors='none', edgecolors='tab:blue')
            plt.scatter([center[0],], [center[1],], marker='o', sizes=[10,])
            plt.plot(rect.ends[..., 0], rect.ends[..., 1], color='orange')
            plt.xlim([10, 50])
            plt.ylim([20, 180])
            plt.xlabel("x")
            plt.ylabel("y")
            plt.show()
        """
        X = np.argwhere(target.array)[..., [1, 0]]
        assert len(X.shape) == 2
        assert X.shape[1] == 2

        X_trans = self.pca.fit_transform(X)

        center = X.mean(axis=0)
        w_vec, h_vec = self.pca.components_

        if self._size_estimator == 'gaussian':
            w, h = norm.fit(X_trans[..., 0])[1] * 3, norm.fit(X_trans[..., 1])[1] * 3
        elif self._size_estimator == 'min_max':
            w, h = X_trans.max(axis=0) - X_trans.min(axis=0)
        else:
            raise ValueError('wrong size estimator')

        rect = Rectangle(w=w, h=h, center=center, w_vec=w_vec, h_vec=h_vec)

        return rect


class SoftRasFitter(RectangleFitter):
    def __init__(self,
                 sigma=0.1,
                 max_iter=5,
                 lr=0.1,
                 iou_target=None,
                 patience=3,
                 min_delta=0,
                 verbose=None,
                 ):
        self._sigma = sigma

        assert isinstance(max_iter, int)
        self._max_iter = max_iter

        self._lr = lr
        self._verbose = verbose

        self._log_iou_target = -torch.log(torch.tensor(iou_target))
        assert patience is None or isinstance(patience, int)
        self._patience = patience
        self._min_delta = min_delta

        self._i = None

        self._rasterizer = None
        self._optimizer = None

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
        self._w = torch.tensor(self._w, dtype=torch.float64, requires_grad=True, device=device)
        self._h = torch.tensor(self._h, dtype=torch.float64, requires_grad=True, device=device)
        self._theta = torch.tensor(0, dtype=torch.float64, requires_grad=True, device=device)

        self._optimizer = RMSprop([self._w, self._h, self._theta], lr=self._lr)

        loss_history = []

        for i in range(self._max_iter):
            self._i = i
            self._rendered = self._rasterizer.rasterize(self._w, self._h, self._theta)
            self._loss = -log_iou(rendered=self._rendered, target=component.array)

            if self._log_iou_target is not None \
                    and self._loss <= self._log_iou_target:
                break

            if self._patience is not None:
                loss_history.append(self._loss.detach().cpu().numpy().tolist())
                if len(loss_history) >= self._patience:
                    if check_loss_history(loss_history, self._min_delta):
                        loss_history.pop(0)
                    else:
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
        return self._w.detach().cpu().numpy()

    @property
    def h(self):
        return self._h.detach().cpu().numpy()

    @property
    def theta(self):
        return self._theta.detach().cpu().numpy()

    @property
    def result(self):
        return self._cc.center, \
               (float(self.w), float(self.h), float(self.theta))


def check_loss_history(history, min_delta=0):
    # return True if continue
    assert len(history) >= 2
    base = history[0]
    for x in history[1:]:
        if base - x > min_delta:
            return True
    return False


def fit_open_points(target: SingleConnectedComponent, fitter=None) -> Rectangle:
    """
    a wrapper for Rectangle Fitter
    :param target:
    :param fitter:
    :return:
    """
    if fitter is None:
        fitter = PCAFitter()
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
