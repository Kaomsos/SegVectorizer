from __future__ import annotations
import typing

if typing.TYPE_CHECKING:
    from segvec.typing_ import SingleConnectedComponent
    from segvec.main_steps.open_points import RectangleFitter
from segvec.main_steps.open_points import SoftRasFitter, PCAFitter

import time
import pickle
import logging

from experiments.common import iou, precision, recall, p_config, Vectorizer


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

        logging.info(f'time: {interval}, iou: {iou_v}, precision: {p_v}, recall: {r_v}')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    with open("../data/refined_seg.pickle", 'rb') as f:
        seg = pickle.load(f)

    vec = Vectorizer(palette_config=p_config)

    open_cc, boundary_cc, room_cc = vec.extract_connected_components(seg)

    # run(open_cc, PCAFitter(size_estimator='min_max'))
    run(open_cc, SoftRasFitter(iou_target=0.8, lr=0.01, patience=3, max_iter=5))
