# %%
from __future__ import annotations
from typing import Tuple

import numpy as np
import torch

from entity_class import Polygon, SingleConnectedComponent
from rasterizer import Base2DPolygonRasterizer as SoftRasterizer
from rasterizer import FixedCenterRectangle2DRasterizer as RectangleRasterizer
from objective import log_iou, boundary, orthogonal
from torch.optim import RMSprop
import importlib


class PolygonVertexIterator:
    def __init__(self, plg: Polygon):
        self._plg = plg

    def __iter__(self):
        self._n_vertices = len(self._plg)
        self._i = 0
        return self

    def __next__(self):
        if self._i >= self._n_vertices:
            raise StopIteration

        self._i += 1
        return self._plg[(self._i - 1) % self._n_vertices]


class PolygonEdgeIterator:
    def __init__(self, plg: Polygon):
        self._plg = plg

    def __iter__(self):
        self._n_edges = len(self._plg)
        self._i = 0
        return self

    def __next__(self):
        if self._i >= self._n_edges:
            raise StopIteration

        self._i += 1

        return self._plg[(self._i - 1) % self._n_edges], \
               self._plg[self._i % self._n_edges]


class PolygonAdjacentEdgesIterator:
    def __init__(self, plg: Polygon):
        self._plg = plg

    def __iter__(self):
        self._n_edges = len(self._plg)
        self._i = 0
        return self

    def __next__(self):
        if self._i >= self._n_edges:
            raise StopIteration

        self._i += 1

        return self._plg[(self._i - 2) % self._n_edges], \
               self._plg[(self._i - 1) % self._n_edges], \
               self._plg[self._i       % self._n_edges]


class VertexReducer:
    def __init__(self,
                 plg: Polygon,
                 delta_a: float = 10,
                 delta_b: float = 10
                 ) -> None:
        self._plg = plg
        self._cur_plg = plg
        self._delta_a = delta_a
        self._delta_b = np.cos(delta_b / 180 * np.pi)
        self._flag = None

    def reduce(self):
        # reduce by one iteration
        self._flag = True
        self.reduce_by_collinear_condition()
        self.reduce_by_adjacent_condition()

    def reduce_by_collinear_condition(self):
        vertices = []
        for adj_es in PolygonAdjacentEdgesIterator(self._plg):
            e1 = adj_es[0] - adj_es[1]
            e2 = adj_es[1] - adj_es[2]
            if not self._is_collinear(e1, e2):
                vertices.append(adj_es[1])
            else:
                self._flag = False
        self._set_polygon(np.array(vertices))

    def reduce_by_adjacent_condition(self):
        vertices = []
        NORMAL = 0
        SKIP = 1
        state = NORMAL
        for e in PolygonEdgeIterator(self._plg):
            p1 = e[0]
            p2 = e[1]
            if state == NORMAL:
                if not self._is_adjacent(p1, p2):
                    state = NORMAL
                    vertices.append(p1)
                else:
                    # d < delta_a
                    self._flag = False
                    state = SKIP
                    p_mid = (p1 + p2) / 2
                    vertices.append(p_mid)
            else:
                # state == SKIP
                if not self._is_adjacent(p1, p2):
                    state = NORMAL
                else:
                    # d < delta_a
                    self._flag = False
                    state = SKIP
                    p_mid = (p1 + p2) / 2
                    vertices.append(p_mid)

        if state == SKIP:
            vertices.pop()

        self._set_polygon(np.array(vertices))

    def _is_collinear(self, e1, e2):
        cos_val = (e1 * e2).sum() / np.sqrt((e1 * e1).sum() * (e2 * e2).sum())
        return cos_val > self._delta_b

    def _is_adjacent(self, p1, p2):
        distance = np.sqrt(((p1 - p2) * (p1 - p2)).sum())
        return distance < self._delta_a

    def _set_polygon(self, vertices: np.ndarray):
        self._plg.set_vertices(vertices)

    @property
    def polygon(self) -> Polygon:
        return self._plg

    @property
    def stop(self) -> bool:
        return self._flag


class CoordinateOptimizer:
    def __init__(self,
                 target: SingleConnectedComponent,
                 sigma: float = 1,
                 lr=0.1,
                 max_iter=5,
                 ) -> None:

        self._cc = target
        self._sigma = sigma
        self._lr = lr
        self._max_iter = max_iter
        self._verbose = None
        self._i = None
        self._P = None
        self._optimizer = None
        self._rasterizer = SoftRasterizer(image_size=self._cc.size,
                                          sigma=self._sigma,
                                          mode='soft euclidean')

    def fit(self, plg: Polygon, verbose=False):
        self._P = plg.torch_tensor
        self._optimizer = RMSprop([self._P], lr=self._lr)
        self._verbose = verbose
        for i in range(self._max_iter):
            self._i = i
            L = self._objective()
            L.backward()
            self._optimizer.step()
            self._hook_after_step()
        plg.set_vertices(self.optimized_result)

    def _rasterize(self, polygon: torch.Tensor):
        self._rendered = self._rasterizer(polygon)
        self._hook_after_render()
        return self._rendered

    def _objective(self) -> torch.Tensor:
        iou = log_iou(self._rasterize(self._P), self._cc.array)
        bo = boundary(self._P, self._cc.boundary)
        orth = orthogonal(self._P)
        return -1 * iou + 5 * bo + 1 * orth

    def plot_soft_ras(self):
        plt = importlib.import_module('.pyplot', 'matplotlib')
        rendered = self._rendered
        rect_img = self._cc
        plt.subplot(1, 2, 1)
        plt.imshow(rendered.detach().numpy())
        plt.title(f"sigma = {self._sigma}, iter = {self._i}")
        plt.subplot(1, 2, 2)
        plt.imshow(rect_img.array + rendered.detach().numpy())
        plt.title(f"IOU")
        plt.show()

    def _hook_after_render(self):
        if self._verbose:
            self.plot_soft_ras()

    def _hook_after_step(self):
        if self._verbose:
            print(f'iter = {self._i}, grad = {self._P.detach().numpy()}')

    @property
    def optimized_result(self):
        return self._P.detach().numpy()


class RectangleOptimizer:
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

    def fit(self,
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

    def fit_return(self,
                   component: SingleConnectedComponent,
                   verbose: bool
                   ):
        self.fit(component, verbose)
        return self.result

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
        return self._cc.center,\
               (float(self.w), float(self.h), float(self.theta))


def alternating_optimize(cc: SingleConnectedComponent,
                         max_iter=5,
                         coord_opt_iter=0
                         ) -> Polygon:
    plg = Polygon(connected_component=cc, tol=5)
    reducer = VertexReducer(plg, delta_a=10)
    opt = CoordinateOptimizer(target=cc, max_iter=0, sigma=10)
    for _ in range(max_iter):
        reducer.reduce()
        if reducer.stop:
            break
        opt.fit(plg, verbose=True)
    return plg


if __name__ == "__main__":
    pass



