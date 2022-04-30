import json
import pickle
import numpy as np
from segvec.entity.wall_center_line import WallCenterLineWithOpenPoints, WallCenterLine
from pathlib import Path
from segvec.geometry import rasterize_polygon
from segvec.main_steps.open_points import PCAFitter
from segvec.main_steps.wall_center_line import alternating_optimize as fit_wall_center_line

import logging
import warnings
from pathlib import Path
import pickle
from common import iou, precision, recall, p_config, Vectorizer

vec = Vectorizer(palette_config=p_config)
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')


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


def get_wall_center_line(seg, config=None):
    open_cc, boundary_cc, room_cc = vec.extract_connected_components(seg)
    rects = vec.get_rectangles(open_cc, PCAFitter(size_estimator='gaussian'))
    vec.set_hyper_parameters_by_rectangles(rects)
    room_contours = vec.get_room_contours(room_cc)

    # logging.info(f'start fitting wall center line {name}')
    wcl = fit_wall_center_line(WallCenterLine(room_contours),
                               delta_x=vec._delta_x,
                               delta_y=vec._delta_y,
                               boundary=boundary_cc,
                               downscale=vec._downscale,
                               max_alt_iter=5,
                               max_coord_iter=100,
                               lr=0.01,
                               patience=3,
                               min_delta=0.,
                               weights=(1, 0, 0),
                               **config,
                               )

    vec.set_widths_of_wcl(wcl, boundary_cc)
    return wcl


if __name__ == '__main__':
    segs = load_wcl_o('exp_data/list_refined_segs.pickle')
    for seg in segs[:1]:
        wcl = get_wall_center_line(seg)
        rast = rasterize_wcl(wcl, seg.shape)
        from segvec.utils import *

        pass
