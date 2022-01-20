from __future__ import annotations
from typing import TYPE_CHECKING, Tuple
if TYPE_CHECKING:
    from ..typing_ import Segment, WallCenterLine
from sklearn.mixture import GaussianMixture
import numpy as np

from ..geometry import rasterize_polygon

def get_two_means(X: np.ndarray):
    X = X.reshape(-1, 1)
    fitter = GaussianMixture(2)
    fitter.fit(X)
    fitter.means_.sort()

    # return [thin, thick]
    return fitter.means_.reshape(-1).tolist()


def rasterize_edge(arr_shape, ends: Segment, width: float):
    p1, p2 = np.array(ends)
    v = p1 - p2
    n = np.array([v[1], -v[0]]) / np.sqrt((v * v).sum())
    delta = n * width / 2
    polygon = np.array([p1, p1, p2, p2]) + np.array([delta, -delta, -delta, delta])
    return rasterize_polygon(arr_shape, polygon)


def ior(target: np.ndarray, rast: np.ndarray):
    target = target.astype(bool)
    rast = rast.astype(bool)
    i = (target & rast).sum()
    r = rast.sum()
    return i / r


def get_width(target: np.ndarray, wcl: WallCenterLine, thin_thick: Tuple[float, float]):
    thin, thick = thin_thick
    arr_shape = target.shape
    for ends in zip(wcl.segments_collection):
        m_thin = ior(target, rasterize_edge(arr_shape, ends, thin))
        m_thick = ior(target, rasterize_edge(arr_shape, ends, thick))

    return