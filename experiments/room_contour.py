from __future__ import annotations
from experiments.common import iou, precision, recall, p_config, Vectorizer
from segvec.vetorizer import fit_room_contour
from segvec.main_steps.open_points import PCAFitter

import time
import pickle
import logging
import pandas as pd

# global variables
record: tuple = ()
logs: list = []
vec = Vectorizer(palette_config=p_config)


def run(segs,
        max_alt_iter=5,
        max_coord_iter=100,
        sigma=10,
        weights=(0, 1, 0),
        lr=0.01,
        patience=5,
        min_delta=0):
    global record
    global logs

    logs = []

    for i, seg in enumerate(segs):
        open_cc, boundary_cc, room_cc = vec.extract_connected_components(seg)
        rects = vec.get_rectangles(open_cc, PCAFitter(size_estimator='gaussian'))
        vec.set_hyper_parameters_by_rectangles(rects)

        logging.info(f'collected {len(room_cc)} items in room {1 + i}')

        for room in room_cc:
            s = time.time_ns()
            contour = fit_room_contour(room,
                                       delta_a=vec._delta_a,
                                       delta_b=vec._delta_b,
                                       max_alt_iter=max_alt_iter,
                                       max_coord_iter=max_coord_iter,
                                       sigma=sigma,
                                       weights=weights,
                                       lr=lr,
                                       patience=patience,
                                       min_delta=min_delta,
                                       )
            e = time.time_ns()

            interval = (e - s) / 1000000
            iou_v = iou(room, contour.numpy_array)
            p_v = precision(room, contour.numpy_array)
            r_v = recall(room, contour.numpy_array)

            record = (interval, iou_v, p_v, r_v)

            after_step_hook()
    after_run_hook()


def after_step_hook():
    global record
    global logs
    global save_path

    interval, iou_v, p_v, r_v = record
    # output to stdout
    logging.info(f'time: {interval}, iou: {iou_v}, precision: {p_v}, recall: {r_v}')
    # store in
    logs.append({'time': interval, 'iou': iou_v, 'precision': p_v, 'recall': r_v})

    with open(save_path, 'a') as f:
        f.write(','.join(map(str, record)) + '\n')


def after_run_hook():
    global logs
    df = pd.DataFrame(logs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    save_path = 'result/room_contour.csv'
    segs_path = "../data/refined_seg.pickle"

    with open(segs_path, 'rb') as f:
        seg = pickle.load(f)

    w_o = {'max_alt_iter': 5,
           'max_coord_iter': 0,
           'lr': 0.01,
           'patience': 5,
           'min_delta': 0,
           'sigma': 10,
           'weights': (0, 0, 0),
           }

    w_iou = {'max_alt_iter': 5,
             'max_coord_iter': 100,
             'lr': 0.01,
             'patience': 5,
             'min_delta': 0,
             'sigma': 10,
             'weights': (1, 0, 0),
             }

    w_boundary = {'max_alt_iter': 5,
                  'max_coord_iter': 100,
                  'lr': 0.01,
                  'patience': 5,
                  'min_delta': 0,
                  'sigma': 10,
                  'weights': (0, 1, 0),
                  }

    w_orth = {'max_alt_iter': 5,
              'max_coord_iter': 100,
              'lr': 0.01,
              'patience': 5,
              'min_delta': 0,
              'sigma': 10,
              'weights': (0, 0, 1),
              }

    full = {'max_alt_iter': 5,
            'max_coord_iter': 100,
            'lr': 0.01,
            'patience': 5,
            'min_delta': 0,
            'sigma': 10,
            'weights': (1, 5, 1),
            }

    run(segs=[seg], **w_o)
