from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple

import torch
from numpy.typing import ArrayLike
from torch import nn, Tensor

from geometry import intersection_given_x_plg, distance_p_to_plg, get_segments, get_bounding_box


class Rasterizer(ABC):
    @abstractmethod
    def rasterize(self, polygons) -> ArrayLike:
        pass


class Base2DPolygonRasterizer(Rasterizer, nn.Module):
    def __init__(self,
                 image_size: Tuple[int, int],
                 sigma: float = 0.001,
                 offset: Tuple[float, float] = (0, 0),
                 scale: float = 1,
                 threshold: float = 30,
                 mode: str | int = "hard assign"
                 ) -> None:
        super().__init__()
        self._image_size = image_size
        self._sigma = torch.as_tensor(sigma)
        self._offset = torch.as_tensor(offset)
        self._scale = torch.as_tensor(scale)
        self._threshold = torch.as_tensor(threshold)
        self._mode = mode

    def forward(self, x):
        # get all segments: non-differentiable
        # indicator variable: non-differentiable
        # min distance: differentiable
        # sigmoid: differentiable
        return self.rasterize(x)

    def rasterize(self, polygon: Tensor) -> ArrayLike:
        # transform by offset and scale
        polygon = torch.as_tensor(polygon, dtype=torch.float64)
        polygon = (polygon - self._offset) * self._scale
        self.update_xy_lim(polygon)

        # scan line algorithm
        scan = self._scan_along_x(polygon)

        # soft ras
        if self._mode == "hard assign":
            m = torch.where(scan, 1, 0)
        elif self._mode == "soft euclidean":
            m = self._compute_prob_map(polygon, scan)
        return m

    def _scan_along_x(self, polygon: Tensor):
        m = self.blank_matrix
        # image-coord 2 array coord
        for i in range(self.x_min, self.x_max + 1):
            if not (0 <= i < self.array_size[1]):
                continue
            # quick exclude
            # compute intersection
            ys = self.get_intersection(i, polygon)

            start = True
            for y_s, y_e in zip(ys[:-1], ys[1:]):
                if start:
                    self.fill_tensor(m, i, y_s, y_e)
                start = not start
        return m

    def _compute_prob_map(self, polygon: Tensor, scan: Tensor):
        indicator = torch.where(scan, 1, -1)
        # d_square_min = torch.zeros(self.array_size, dtype=torch.float64)
        map_ = torch.zeros(self.array_size, dtype=torch.float64)
        for i in range(self.array_size[0]):
            if i < self.y_min - self._threshold \
               or i > self.y_max + self._threshold:
                continue

            for j in range(self.array_size[1]):
                if j < self.x_min - self._threshold \
                   or j > self.x_max + self._threshold:
                    continue

                d_square_min = \
                    self.square_euclidean_p_to_plg(point=(j, i), polygon=polygon)
                map_[i, j] = 0 if d_square_min >= self._threshold * self._threshold and indicator[i, j] < 0 \
                             else torch.sigmoid(indicator[i, j] * d_square_min / self._sigma)
        return map_

    @staticmethod
    def get_intersection(x, polygon):
        return intersection_given_x_plg(x, polygon)

    @staticmethod
    def square_euclidean_p_to_plg(point: Tensor | Tuple[int, int], polygon: Tensor):
        return distance_p_to_plg(point, polygon, square=True)

    def fill_tensor(self, ts, x, y_s, y_e):
        s = y_s.int()
        e = y_e.int() + 1
        s, e = (e, s) if (s > e) else (s, e)
        s = self._clip_to_image_height(s)
        e = self._clip_to_image_height(e)
        ts[s: e, x] = True

    @staticmethod
    def get_segments(polygon: Tensor) -> Tuple[Tensor, Tensor]:
        return get_segments(polygon)

    @staticmethod
    def get_bounding_box(points: Tensor) -> Tuple[Tensor, Tensor]:
        return get_bounding_box(points)

    def update_xy_lim(self, polygon):
        lt, rb = self.get_bounding_box(polygon)
        lt = torch.floor(lt).int()
        rb = torch.floor(rb).int() + 1
        self.x_min, self.y_min = lt
        self.x_max, self.y_max = rb

    def _clip_to_image_height(self, i):
        i = i if i >= 0 else 0
        i = i if i <= self.array_size[0] else self.array_size[0]
        return i

    @property
    def array_size(self):
        # image-coordinate 2 array-coordinate
        return self._image_size[1], self._image_size[0]

    @property
    def blank_matrix(self) -> Tensor:
        m = torch.full(self.array_size, fill_value=False, dtype=torch.bool)
        return m


class BoundingBox2DRasterizer(Base2DPolygonRasterizer):
    def rasterize(self, diagonal: Tensor) -> ArrayLike:
        p0 = diagonal[0]
        p2 = diagonal[1]
        p1 = torch.stack([p0[0], p2[1]])
        p3 = torch.stack([p2[0], p0[1]])
        polygon = torch.stack([p0,
                               p1,
                               p2,
                               p3], dim=0)
        return super(BoundingBox2DRasterizer, self).rasterize(polygon)


class FixedCenterRectangle2DRasterizer(Base2DPolygonRasterizer):
    def forward(self, x):
        w = x[0]
        h = x[1]
        theta = x[2]
        return self.rasterize(w, h, theta)

    def __init__(self,
                 image_size: Tuple[int, int],
                 center: Tuple[float, float],
                 sigma: float = 0.001,
                 offset: Tuple[float, float] = (0, 0),
                 scale: float = 1,
                 mode: str | int = "hard assign"
                 ) -> None:
        super(FixedCenterRectangle2DRasterizer, self).__init__(image_size=image_size,
                                                               sigma=sigma,
                                                               offset=offset,
                                                               scale=scale,
                                                               mode=mode)
        self._center = torch.as_tensor(center)

    def rasterize(self, w: Tensor, h: Tensor, theta: Tensor) -> ArrayLike:
        offset = torch.stack([w, h], dim=0)

        p0 = self._center + offset / 2
        p2 = self._center - offset / 2
        p1 = torch.stack([p0[0], p2[1]], dim=0)
        p3 = torch.stack([p2[0], p0[1]], dim=0)
        rect = torch.cat([p0[None, ...],
                          p1[None, ...],
                          p2[None, ...],
                          p3[None, ...]], dim=0)

        R = torch.stack([
                            torch.stack([torch.cos(theta), -torch.sin(theta)]),
                            torch.stack([torch.sin(theta), torch.cos(theta)])
                        ])
        polygon = R @ (rect - self._center) + self._center
        return super(FixedCenterRectangle2DRasterizer, self).rasterize(polygon)
