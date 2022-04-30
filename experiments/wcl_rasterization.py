import json
import pickle
import numpy as np
from segvec.entity.wall_center_line import WallCenterLineWithOpenPoints
from pathlib import Path
from segvec.geometry import rasterize_polygon


def get_polygon_of_wall(edge, width) -> np.ndarray:
    p1, p2 = edge
    vec = p1 - p2
    norm_vec = np.array([vec[1], -vec[0]]) / np.sqrt((vec * vec).sum())
    delta = norm_vec * width / 2
    polygon = np.vstack([p1, p1, p2, p2]) + np.vstack([delta, -delta, -delta, delta])
    return polygon


def rasterize_wcl(wcl: WallCenterLineWithOpenPoints, shape) -> np.ndarray:
    rast = np.full(shape, False, dtype=bool)
    for e in wcl.edges:
        v1, v2 = e
        p1, p2 = wcl.get_coordinate_by_v(v1), wcl.get_coordinate_by_v(v2)
        width = wcl.get_width_by_e(e)
        polygon = get_polygon_of_wall((p1, p2), width)
        wall_rast = rasterize_polygon(shape, polygon)
        rast |= wall_rast
    return rast


def load_wcl_o(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


if __name__ == '__main__':
    shape = 500, 500
    wcl_o: WallCenterLineWithOpenPoints = load_wcl_o('../experiments/raw_demo/2_1k8_wcl.pickle')

    rast = rasterize_wcl(wcl_o, shape)
    from segvec.utils import *
    pass
