from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..typing_ import SingleConnectedComponent, Contour

import importlib

import numpy as np
import torch
from torch.optim import RMSprop

from ..entity.polygon import Polygon
from ..optimize.objective import log_iou, boundary, orthogonal
from ..optimize.softras import Base2DPolygonRasterizer as SoftRasterizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
                 weights=(0, 0, 0),
                 lr=0.1,
                 max_iter=5,
                 patience=3,
                 min_delta=0,
                 iou_target=None,
                 ) -> None:

        self._cc = target
        self._sigma = sigma
        self._w1, self._w2, self._w3 = weights
        self._lr = lr

        self._max_iter = max_iter

        self._patience = patience
        self._min_delta = min_delta

        self._verbose = None
        self._i = None
        self._P = None
        self._optimizer = None
        self._rasterizer = SoftRasterizer(image_size=self._cc.size,
                                          sigma=self._sigma,
                                          mode='soft euclidean')

    def fit(self, plg: Polygon, verbose=False):
        """
        fit the polygon and return
        :param plg:
        :param verbose:
        :return:
        """
        self._P = plg.torch_tensor.clone().to(device).detach().requires_grad_(True)
        self._optimizer = RMSprop([self._P], lr=self._lr)
        self._verbose = verbose

        loss_history = []
        for i in range(self._max_iter):
            self._i = i
            L = self._objective()

            loss_history.append(L.detach().cpu().numpy())
            if len(loss_history) >= self._patience:
                if check_loss_history(loss_history, min_delta=self._min_delta):
                    loss_history.pop(0)
                else:
                    break
            L.backward()

            self._optimizer.step()
            self._hook_after_step()
        # update the polygon
        plg.set_vertices(self.optimized_result)

    def _rasterize(self, polygon: torch.Tensor):
        self._rendered = self._rasterizer.rasterize(polygon)
        self._hook_after_render()
        return self._rendered

    def _objective(self) -> torch.Tensor:
        if self._w1 != 0:
            rast = self._rasterize(self._P)
            target = torch.tensor(self._cc.array, device=device)
            iou = log_iou(rast, target)
        else:
            iou = torch.tensor(0., requires_grad=True)

        if self._w2 != 0:
            bo = boundary(self._P, self._cc.boundary)
        else:
            bo = torch.tensor(0., requires_grad=True)

        if self._w3 != 0:
            orth = orthogonal(self._P)
        else:
            orth = torch.tensor(0., requires_grad=True)

        return -self._w1 * iou + self._w2 * bo + self._w3 * orth

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
        return self._P.detach().cpu().numpy()


def check_loss_history(history, min_delta=0):
    # return True if continue
    assert len(history) >= 2
    base = history[0]
    for x in history[1:]:
        if base - x > min_delta:
            return True
    return False


def alternating_optimize(cc: SingleConnectedComponent,
                         delta_a=10,
                         delta_b=10,
                         max_alt_iter=5,
                         max_coord_iter=0,
                         sigma=10,
                         weights=(0, 0, 0),
                         lr=0.01,
                         patience=3,
                         min_delta=0,
                         ) -> Contour:
    plg = Polygon(connected_component=cc, tol=5)
    reducer = VertexReducer(plg, delta_a=delta_a, delta_b=delta_b)
    opt = CoordinateOptimizer(target=cc,
                              sigma=sigma,
                              weights=weights,
                              max_iter=max_coord_iter,
                              lr=lr,
                              patience=patience,
                              min_delta=min_delta)
    for _ in range(max_alt_iter):
        reducer.reduce()
        if reducer.stop:
            break
        opt.fit(plg, verbose=False)
    return plg
