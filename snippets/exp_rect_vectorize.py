from __future__ import  annotations
import typing
if typing.TYPE_CHECKING:
    from segvec.typing_ import SingleConnectedComponent
    from segvec.main_steps.open_points import RectangleFitter
from segvec.vetorizer import Vectorizer, PaletteConfiguration
from segvec.geometry import rasterize_polygon
from segvec.main_steps.open_points import fit_open_points, PCAFitter, SoftRasFitter

import numpy as np
import time
import pickle


def _iou(target: np.ndarray, rast: np.ndarray, mask: np.ndarray = None):
    # intersection over right
    target = target.astype(bool)
    rast = rast.astype(bool)
    if mask is not None:
        mask = mask.astype(bool)
        target &= mask

    i = (target & rast).sum()
    u = (rast | target).sum()
    return i / u


def _precision(target: np.ndarray, rast: np.ndarray, mask: np.ndarray = None):
    # pixel precision
    rast = rast.astype(bool)
    target = target.astype(bool)
    if mask is not None:
        mask = mask.astype(bool)
        target &= mask

    i = (target & rast).sum()
    rast_n = rast.sum()
    return i / rast_n


def _recall(target: np.ndarray, rast: np.ndarray, mask: np.ndarray = None):
    # pixel recall
    rast = rast.astype(bool)
    target = target.astype(bool)
    if mask is not None:
        mask = mask.astype(bool)
        target &= mask

    i = (target & rast).sum()
    target_n = target.sum()
    return i / target_n


def iou(target: SingleConnectedComponent, approx: np.ndarray):
    rast = rasterize_polygon(arr_shape=target.shape, polygon=approx)
    return _iou(target.array, rast)


def precision(target: SingleConnectedComponent, approx: np.ndarray):
    rast = rasterize_polygon(arr_shape=target.shape, polygon=approx)
    return _precision(target.array, rast)


def recall(target: SingleConnectedComponent, approx: np.ndarray):
    rast = rasterize_polygon(arr_shape=target.shape, polygon=approx)
    return _recall(target.array, rast)


def run(open_cc_list: typing.List[SingleConnectedComponent],
        fitter: RectangleFitter):
    for open_cc in open_cc_list:
        s = time.time_ns()
        rect = fitter.fit(open_cc)
        e = time.time_ns()

        interval = (e - s) / 1000000
        iou_v = iou(open_cc, rect.polygon)
        p_v = precision(open_cc, rect.polygon)
        r_v = recall(open_cc, rect.polygon)

        print(f'time: {interval}, iou: {iou_v}, precision: {p_v}, recall: {r_v}')


if __name__ == "__main__":
    with open("../data/refined_seg.pickle", 'rb') as f:
        seg = pickle.load(f)
    p = {
        '厨房': -1,
        '阳台': 1,
        '卫生间': 2,
        '卧室': 3,
        '客厅': 4,
        '墙洞': 5,
        '玻璃窗': 6,
        '墙体': 7,
        '书房': 8,
        '储藏间': 9,
        '门厅': 10,
        '其他房间': 11,
        '未命名': 12,
        '客餐厅': 13,
        '主卧': 14,
        '次卧': 15,
        '露台': 16,
        '走廊': 17,
        '设备平台': 18,
        '储物间': 19,
        '起居室': 20,
        '空调': 21,
        '管道': 22,
        '空调外机': 23,
        '设备间': 24,
        '衣帽间': 25,
        '中空': 26
    }

    p_config = PaletteConfiguration(p,
                                    add_door=('墙洞',),
                                    add_window=('玻璃窗',),
                                    add_boundary=('墙体',),
                                    add_room=('厨房', '阳台', '卫生间', '卧室', '客厅', '书房',
                                              '储藏间', '门厅', '其他房间',
                                              '未命名', '客餐厅', '主卧', '次卧', '露台', '走廊',
                                              '设备平台', '储物间', '起居室', '空调', '管道',
                                              '空调外机', '设备间', '衣帽间', '中空'),
                                    )

    vec = Vectorizer(palette_config=p_config)

    ##############################################
    # vectorization
    ##############################################
    # Init
    open_cc, boundary_cc, room_cc = vec.extract_connected_components(seg)

    run(open_cc, PCAFitter(size_estimator='min_max'))
    # run(open_cc, SoftRasFitter(iou_target=0.8, lr=0.01, patience=3, max_iter=5))

