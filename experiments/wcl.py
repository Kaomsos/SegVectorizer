from __future__ import annotations
from common import iou, precision, recall, p_config, Vectorizer
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    segs_path = "../data/refined_seg.pickle"

    save_path = 'result/room_contour.csv'
