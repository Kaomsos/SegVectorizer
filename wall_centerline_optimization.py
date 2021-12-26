from __future__ import annotations
from typing import Dict

import matplotlib.pyplot as plt
import torch
from torch.optim import RMSprop

from entity_class import WallCenterLine, SingleConnectedComponent, SemiIdentityMapping
from sklearn.neighbors import kneighbors_graph
from geometry import distance_p_to_segments
from objective import center, nearby, alignment
import numpy as np


class VertexReducer:
    def __init__(self,
                 wcl: WallCenterLine,
                 delta_x: float = 10,
                 delta_y: float = 10,
                 ) -> None:
        self._wcl = wcl
        self._delta_x = delta_x
        self._delta_y = delta_y
        self._flag: bool = None

    def reduce(self):
        self._flag = True
        self.reduce_by_condition_x(delta_x=self._delta_x)
        self.reduce_by_condition_y(delta_y=self._delta_y)

    def reduce_by_condition_x(self, delta_x: float):
        g = kneighbors_graph(self._wcl.V, 3, mode='distance', include_self=False)
        g = (g > 0).toarray() & (g < delta_x).toarray()
        edges = np.argwhere(np.triu(g | g.T)).tolist()
        i2v = self._wcl.i2v
        after_merge = SemiIdentityMapping()
        for i, j in edges:
            i_after = after_merge(i)
            j_after = after_merge(j)
            self._wcl.merge_vertices(i2v[i_after], i2v[j_after])
            after_merge[j_after] = i_after
            self._flag = False

    def reduce_by_condition_y(self, delta_y: float):
        ps = self._wcl.V
        i2v = self._wcl.i2v
        for i, p in enumerate(ps):
            segs = self._wcl.segments_collection
            i2e = self._wcl.i2e
            ds = distance_p_to_segments(p, segs)

            v = i2v[i]
            for i_ in np.argwhere(ds < delta_y).reshape(-1):
                e = i2e[int(i_)]
                if v not in e:
                    self._wcl.append_vertex_to_edge(v, e)
                    self._flag = False

    @property
    def stop(self):
        return self._flag


class CoordinateOptimizer:
    def __init__(self,
                 wcl: WallCenterLine,
                 max_iter: int = 5,
                 lr: float = 0.1,
                 downscale: int = 4,
                 ) -> None:
        self._center = center
        self._nearby = nearby
        self._alignment = alignment
        self._wcl = wcl
        self._dtype = torch.float64
        self.V = None
        self._target = None
        self._downscale = downscale
        self._optimizer = None
        self._w1 = 1
        self._w2 = 2
        self._w3 = 1
        self._max_iter = max_iter
        self._i = None
        self._verbose = False
        self._lr = lr

    def fit(self,
            target: SingleConnectedComponent | np.ndarray,
            verbose: bool = False
            ):
        self._verbose = verbose
        self._target = target
        self._fit_init(target)
        for i in range(self._max_iter):
            self._i = i
            L = self._objective()
            L.backward()
            self._optimizer.step()
            self._hook_after_step()
        self._update_wlc()

    def _fit_init(self, target):
        #################
        # variable
        self.V = torch.tensor(self._wcl.V, dtype=self._dtype, requires_grad=True)

        ##################
        # constants
        ##################
        # coordinates
        self.P = torch.as_tensor(self._wcl.P, dtype=self._dtype)
        # binary image
        self.boundaries = getattr(target, 'array', target)
        # matrices
        self.S, self.E = self._wcl.segments_matrix
        self.S = torch.as_tensor(self.S, dtype=self._dtype)
        self.E = torch.as_tensor(self.E, dtype=self._dtype)
        self.EDGES = self.S - self.E

        self.L = torch.as_tensor(self._wcl.L, dtype=self._dtype)

        ##################
        self._optimizer = RMSprop([self.V], lr=self._lr)

    def _objective(self) -> torch.Tensor:
        # compute objectives
        center = self._center((self.S @ self.V, self.E @ self.V),
                              self.boundaries,
                              downscale_factor=self._downscale)
        nearby = self._nearby(self.V, self.P, self.L)
        alignment = self._alignment(self.EDGES @ self.V)

        loss = self._w1 * center + self._w2 * nearby + self._w3 * alignment
        return loss

    def _update_wlc(self):
        self._wcl.set_current_coordinates(self.optimized_result)

    def _hook_after_step(self):
        if self._verbose:
            print(f"iter = {self._i}, grad = {self.V.grad}")
            self._update_wlc()
            self.plot()

    def plot(self):
        from utils import plot_wall_center_lines
        plt.imshow(1 - self._target, cmap='gray')
        plot_wall_center_lines(self._wcl)
        plt.show()

    @property
    def optimized_result(self) -> np.ndarray:
        return self.V.detach().numpy()


def alternating_optimize(wcl: WallCenterLine,
                         boundary: np.ndarray,
                         delta_x: float = 10,
                         delta_y: float = 10,
                         downscale: int = 4,
                         max_alt_iter: int = 5,
                         max_coord_iter: int = 10
                         ) -> WallCenterLine:
    reducer = VertexReducer(wcl, delta_x, delta_y)
    optimizer = CoordinateOptimizer(wcl, downscale=downscale, max_iter=max_coord_iter, lr=0.1)

    # iterating for at most max_iter
    for i in range(max_alt_iter):
        reducer.reduce()
        if reducer.stop:
            break
        optimizer.fit(boundary, verbose=False)

    return wcl
