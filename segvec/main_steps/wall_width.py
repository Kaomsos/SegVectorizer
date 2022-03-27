from __future__ import annotations
from typing import TYPE_CHECKING, Tuple, Dict
if TYPE_CHECKING:
    from ..typing_ import Segment, WallCenterLine
from sklearn.mixture import GaussianMixture
import numpy as np

from ..geometry import rasterize_polygon, get_bounding_box


def get_two_means(X: np.ndarray):
    """
    get widths of thin and thick walls with GaussianMixture model
    :param X:
    :return:
    """
    X = X.reshape(-1, 1)
    fitter = GaussianMixture(2)
    fitter.fit(X)
    fitter.means_.sort()

    # return [thin, thick]
    return fitter.means_.reshape(-1).tolist()


def rasterize_by_rec_center(arr_shape, center_line: Segment, width: float):
    p1, p2 = np.array(center_line)
    vec = p1 - p2
    normal_vec = np.array([vec[1], -vec[0]]) / np.sqrt((vec * vec).sum())
    delta = normal_vec * width / 2
    polygon = np.vstack([p1, p1, p2, p2]) + np.vstack([delta, -delta, -delta, delta])
    return rasterize_polygon(arr_shape, polygon)


def rasterize_by_rec_edge(arr_shape, edge: Segment, width: float, reverse=False):
    p1, p2 = np.array(edge)
    vec = p1 - p2
    normal_vec = np.array([vec[1], -vec[0]]) / np.sqrt((vec * vec).sum())
    delta = normal_vec * width
    if reverse:
        delta = -delta
    polygon = np.vstack([p1, p1, p2, p2]) + np.vstack([0, delta, delta, 0])
    return rasterize_polygon(arr_shape, polygon)


def iou(target: np.ndarray, rast: np.ndarray, mask: np.ndarray = None):
    # intersection over right
    target = target.astype(bool)
    rast = rast.astype(bool)
    if mask is not None:
        mask = mask.astype(bool)
        target &= mask

    i = (target & rast).sum()
    u = (rast | target).sum()
    return i / u


def get_width(target: np.ndarray, wcl: WallCenterLine, thin_thick: Tuple[float, float], boundary=None):
    thin, thick = thin_thick
    arr_shape = target.shape
    widths = []
    clses = []
    for ends in zip(*wcl.segments_collection):
        m_thin = iou(target, rasterize_by_rec_center(arr_shape, ends, thin))
        m_thick = iou(target, rasterize_by_rec_center(arr_shape, ends, thick))
        solver = WallWidthSolver(target, ends, boundary, threshold=(thin + thick) / 2)
        clses.append(solver.solve())
        widths.append(thin if m_thin > m_thick else thick)
    return widths


class WallWidthSolver:
    def __init__(self,
                 target: np.ndarray,
                 edge: Segment,
                 boundary: Tuple[float, float],
                 threshold: float
                 ) -> None:
        self._target = target
        self._arr_shape = target.shape

        self._p1, self._p2 = np.array(edge)
        vec = self._p1 - self._p2
        self._normal_vec = np.array([vec[1], -vec[0]]) / np.sqrt((vec * vec).sum())

        self._min, self._max = boundary
        self._min, self._max = round(self._min), round(self._max)
        self._threshold = round(threshold)

        self._mask = rasterize_by_rec_center(self._arr_shape, edge, self._max * 1.5)

        self._w1 = None
        self._w2 = None

        self._objective = None
        self._f: Dict[int, float] = {0: 0}
        self._ldf: Dict[int, float] = {}
        self._rdf: Dict[int, float] = {}

    def _rasterize(self, width: float, reverse=False) -> np.ndarray:
        delta = self._normal_vec * width
        if reverse:
            delta = -delta
        polygon = np.vstack([self._p1, self._p1, self._p2, self._p2]) + np.vstack([delta*0, delta, delta, delta*0])
        return rasterize_polygon(self._arr_shape, polygon)

    def _iou(self, width: float) -> float:
        return iou(self._target,
                   self._rasterize(width, reverse=False),
                   self._mask)

    def _reverse_iou(self, width: float) -> float:
        return iou(self._target,
                   self._rasterize(width, reverse=True),
                   self._mask)

    def solve(self):
        self._objective = self._iou
        threshold = self._threshold
        case = self._inspect_threshold(threshold)
        if case == "continue":
            lb = 0
            self._w1 = self._opt_w((lb, threshold))

        elif case == "thick":
            return "thick"

        elif case == "optimum":
            self._w1 = self._threshold

        else:
            raise ValueError("wrong case")

        self._clear_f()

        self._objective = self._reverse_iou
        threshold = self._threshold - self._w1
        case = self._inspect_threshold(threshold)
        if case == "continue":
            lb = (self._min - self._w1) if (self._min - self._w1) > 0 else 0
            self._w2 = self._opt_w((lb, threshold))
            if self._w1 + self._w2 > self._threshold:
                return "thick"
            else:
                return "thin"

        elif case == "thick":
            return "thick"

        elif case == "optimum":
            if self._f[threshold + 1] > self._f[threshold - 1]:
                return "thick"
            else:
                return "thin"

        else:
            raise ValueError("wrong case")

    def _inspect_threshold(self, threshold):
        if self._is_increasing_on(threshold) and self._is_decreasing_on(threshold):
            # this case conflicts with the presumption
            # that objective function decreases after increases
            raise ValueError("odd point")
        elif self._is_increasing_on(threshold):
            return "thick"
        elif self._is_decreasing_on(threshold):
            return "continue"
        else:
            return "optimum"

    def _opt_w(self, range_: Tuple[int, int]):
        l, u = range_
        m = round((l + u) / 2)
        assert self._in_range(l) and self._in_range(u)

        l_df = self._derivative(l)
        u_df = self._derivative(u)
        m_df = self._derivative(m)

        if l_df == 0 and u_df == 0:
            return 0
        elif l_df == 0:
            return l
        elif u_df == 0:
            return u
        elif m_df == 0:
            return m
        else:
            if l_df <= 0 and l == 0:
                return 0
            elif u_df >= 0 and u == self._max:
                return self._max
            elif m_df < 0 <= l_df:
                return self._opt_w((l, m))
            elif u_df <= 0 < m_df:
                return self._opt_w((m, u))

    def _is_increasing_on(self, x):
        if self._right_derivative(x) is None:
            return False
        return self._right_derivative(x) > 0

    def _is_decreasing_on(self, x):
        if self._left_derivative(x) is None:
            return False
        return self._left_derivative(x) < 0

    def _derivative(self, x) -> float:
        ldf = self._left_derivative(x)
        rdf = self._right_derivative(x)

        if rdf is None and ldf is None:
            raise ValueError('no derivative')
        elif rdf is None:
            df = ldf
        elif ldf is None:
            df = rdf
        elif rdf * ldf > 0:
            df = (ldf + rdf) / 2
        else:
            df = 0

        return df

    def _left_derivative(self, x) -> float:
        self._log_f(x)
        self._log_f(x - 1)

        ldf = self._ldf.get(x, None)

        return ldf

    def _right_derivative(self, x) -> float:
        self._log_f(x)
        self._log_f(x + 1)

        rdf = self._rdf.get(x, None)

        return rdf

    def _clear_f(self):
        self._f = {0: 0}
        self._ldf: Dict[int, float] = {}
        self._rdf: Dict[int, float] = {}

    def _log_f(self, x):
        x = round(x)
        if x not in self._f.keys() and self._in_range(x):
            self._f[x] = self._objective(x)
            if x + 1 in self._f.keys():
                d = self._f[x + 1] - self._f[x]
                self._rdf[x] = d
                self._ldf[x + 1] = d
            if x - 1 in self._f.keys():
                d = self._f[x] - self._f[x - 1]
                self._ldf[x] = d
                self._rdf[x - 1] = d

    def _in_range(self, x):
        return 0 <= x <= self._max
