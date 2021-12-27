from __future__ import annotations
from typing import Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    from ..typing_ import Coordinate2D, Length, Radius, Vector2D

import numpy as np
import torch
from skimage.measure import approximate_polygon, find_contours

from .image import SingleConnectedComponent


class Polygon:
    def __init__(self,
                 connected_component: SingleConnectedComponent = None,
                 arr: np.ndarray = None,
                 vertices: np.ndarray = None,
                 tol: float | int = None,
                 ) -> None:
        """

        :param arr: np.ndarray that represents a binary image
        :param vertices: np.ndarray that represents a list vertices
        """
        self._tol = tol if tol is not None else 30
        # extract polygon's vertices
        # from connected pixels target
        vertices_ = None
        if connected_component is not None:
            assert isinstance(connected_component, SingleConnectedComponent)
            vertices_ = approximate_polygon(connected_component.boundary, tolerance=self._tol)

        elif arr is not None:
            contours = find_contours(arr)    # delete the last point
            assert len(contours) <= 1, 'more than one connected target are detected'
            assert len(contours) > 0, 'no connected target is detected'
            vertices_ = approximate_polygon(contours[0], tolerance=self._tol)
            vertices_ = vertices_[..., [1, 0]]
        # directly assign vertices
        elif vertices_ is None:
            vertices_ = vertices

        self._vertices = None
        self.set_vertices(vertices_)
        assert self._vertices is not None, 'no definition of Polygon'

    def __len__(self):
        return self._vertices.shape[0] - 1

    def __getitem__(self, item):
        assert isinstance(item, int)
        assert item < len(self)

        return self._vertices[item]

    def set_vertices(self, vertices: np.ndarray):
        assert len(vertices) > 0, 'empty array'
        assert len(vertices.shape) == 2, 'not a 2d array'
        assert vertices.shape[1] == 2, 'not a array of 2d point'

        # constraints that self._vertices are "closed"
        # IMPROVE: replace indexing implementation
        if (vertices[0] != vertices[-1]).any():
            idx = list(range(vertices.shape[0]))
            self._vertices = vertices[idx + [0]]
        else:
            self._vertices = vertices

    @property
    def plot_x(self):
        return self._vertices[..., 0]

    @property
    def plot_y(self):
        return self._vertices[..., 1]

    @property
    def torch_tensor(self):
        return torch.tensor(self._vertices[:-1], requires_grad=True)

    @property
    def numpy_array(self):
        return self._vertices[:-1]


class Rectangle:
    """
     a vector representation of any rectangle on a 2D plane
    """
    def __init__(self,
                 center: Coordinate2D,
                 w: Length,
                 h: Length,
                 theta: Radius = None,
                 w_vec: Vector2D = None,
                 h_vec: Vector2D = None,
                 tag=None,
                 ) -> None:
        self.center = np.array(center)
        self.w = w
        self.h = h
        self.theta = theta
        self.R = None
        self.w_vec = np.array(w_vec)
        self.w_vec = self.w_vec / np.sqrt((self.w_vec * self.w_vec).sum())
        self.h_vec = np.array(h_vec)
        self.h_vec = self.h_vec / np.sqrt((self.h_vec * self.h_vec).sum())

        self.has_theta: bool = theta is not None
        if self.has_theta:
            self.R: np.ndarray = np.array([[np.cos(theta), -np.sin(theta)],
                                           [np.sin(theta), np.cos(theta)]])

        self.has_vec: bool = (w_vec is not None) and (h_vec is not None)

        assert self.has_theta or self.has_vec

        self.tag = tag

    @property
    def ends(self) -> Tuple[Coordinate2D, Coordinate2D] | np.ndarray:
        assert self.has_theta or self.has_vec

        if self.has_vec:
            direction = self.w_vec if self.w > self.h else self.h_vec

        elif self.has_theta:
            direction = np.array([1, 0]) if self.w > self.h else np.array([0, 1])
            direction = self.R @ direction
        else:
            raise ValueError("no enough parameters to define a recrangle")

        delta = max(self.w, self.h) / 2

        return self.center + np.array([direction, -direction]) * delta

