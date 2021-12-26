from __future__ import annotations
import importlib
from typing import Tuple, Optional, List

import numpy as np
import torch
from skimage.measure import find_contours


class BinaryImage:
    def __init__(self,
                 bin_array: np.array,
                 ) -> None:
        self.array = bin_array
        self.pos = 1

    @property
    def size(self) -> Tuple[int, int]:
        return self.array.shape[1], self.array.shape[0]

    @property
    def shape(self) -> Tuple[int, int]:
        return self.array.shape

    @property
    def bbox(self):
        indices: np.ndarray = np.argwhere(self.array == self.pos)
        # array-coordinate to pixel-coordinates
        y_min, x_min = np.min(indices, axis=0).tolist()
        y_max, x_max = np.max(indices, axis=0).tolist()
        return (x_min, y_min), (x_max, y_max)

    @property
    def center(self) -> Tuple[float, float]:
        # array-coordinate
        (x_min, y_min), (x_max, y_max) = self.bbox
        return (x_min + x_max) / 2, (y_min + y_max) / 2

    @property
    def width_height(self) -> Tuple[float, float]:
        (x_min, y_min), (x_max, y_max) = self.bbox
        width = x_max - x_min
        height = y_max - y_min
        return width, height


class SingleConnectedComponent(BinaryImage):
    def __new__(cls,
                bin_array: np.ndarray,
                threshold: float = 10
                ) -> Optional[SingleConnectedComponent]:
        inst = super().__new__(cls)
        super(SingleConnectedComponent, inst).__init__(bin_array)

        # filter trivial pixels
        if threshold > 1:
            (x_min, y_min), (x_max, y_max) = inst.bbox
            if max(x_max - x_min, y_max - y_min) < threshold:
                return None

        # init contours
        inst._init_contours(filter_factor=0.9)
        contours = inst.contours

        # try to return a SingleConnectedComponent instance
        if len(contours) == 1:
            return inst
        elif len(contours) > 1:
            warnings = importlib.import_module('warnings')
            warnings.warn("more than one connected components are found, "
                          "but only the first components are considered")
            return inst
        else:
            # construction failed,
            # return None
            return

    def __init__(self, bin_array, threshold=10):
        pass

    def _init_contours(self, filter_factor=0.9):
        # core function
        contours = find_contours(self.array, fully_connected='high')    # coord = array

        # IMPORTANT:
        # transform from array-coord to pixel-coord
        for i in range(len(contours)):
            contours[i] = contours[i][..., [1, 0]]

        # filter contours that's trivial
        (x_min, y_min), (x_max, y_max) = self.bbox
        threshold = filter_factor * min(x_max - x_min, y_max - y_min)
        contours = self.filter_contours(contours, threshold)

        self.contours = contours

    @staticmethod
    def filter_contours(contours: List[np.array], threshold: float = 10):
        # solving the problem of circular import
        mod = importlib.import_module("geometry")
        get_bounding_box = getattr(mod, 'get_bounding_box')

        # define an inner function
        def is_trivial(contour):
            lt, rb = get_bounding_box(torch.as_tensor(contour))
            delta = torch.abs(lt - rb)

            # a contour is trivial iff
            # its x-interval ans y-interval
            # are both smaller than the threshold
            return (delta < threshold).all()

        filtered = [c for c in contours if not is_trivial(c)]
        return filtered

    @property
    def boundary(self) -> np.ndarray:
        return self.contours[0][: -1]