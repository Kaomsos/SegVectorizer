from __future__ import annotations
from typing import List, TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from typing_ import SingleConnectedComponent, Polygon
import unittest
from PIL import Image
import numpy as np


class TestVectorizer(unittest.TestCase):
    def _init(self, path='../data/flat_1.png'):
        from vetorizer import Vectorizer, PaletteConfiguration
        from utils import palette

        self.img = Image.open(path)

        p_config = PaletteConfiguration()
        p_config.add_open("door&window")
        for item in ['bathroom/washroom',
                     'livingroom/kitchen/dining add_room',
                     'bedroom',
                     'hall',
                     'balcony',
                     'closet']:
            p_config.add_room(item)
        p_config.add_boundary("wall")

        self.assertTrue(isinstance(p_config.rooms, set))
        for e in p_config.rooms:
            self.assertTrue(isinstance(e, str))
        self.assertTrue(isinstance(p_config.opens, set))
        for e in p_config.opens:
            self.assertTrue(isinstance(e, str))
        self.assertTrue(isinstance(p_config.boundaries, set))
        for e in p_config.boundaries:
            self.assertTrue(isinstance(e, str))

        self.vectorizer = Vectorizer(palette=palette, palette_config=p_config)

    def test_vectorizer_get_components(self):
        from entity_class import SingleConnectedComponent

        self._init(path='../data/flat_1.png')

        segmentation = np.array(self.img)

        c1, c2, c3 = self.vectorizer._extract_connected_components(segmentation)
        self.assertTrue(isinstance(c2, np.ndarray))
        for ccs in [c1, c3]:
            self.assertTrue(isinstance(ccs, list))
            for cc in ccs:
                self.assertTrue(isinstance(cc, SingleConnectedComponent))

    def test_vectorize_get_wall_center_lines(self):
        from typing_ import Contour, WallCenterLine

        self._init(path='../data/flat_1.png')

        segmentation = np.array(self.img)

        opens, boundary, rooms = self.vectorizer._extract_connected_components(segmentation)

        contours = [self.vectorizer._get_room_contour(cc) for cc in rooms]

        ###################
        # viz: show contours
        self.plot_contours_against_image(contours)

        for c in contours:
            self.assertTrue(isinstance(c, Contour))

        wcl = self.vectorizer._get_wall_center_line(contours, boundary)
        self.assertTrue(isinstance(wcl, WallCenterLine))

        pass

    def test_vectorize(self):
        import pickle
        from utils import plot_wcl_against_target
        self._init(path='../data/flat_1.png')

        segmentation = np.array(self.img)

        opens, boundary, rooms = self.vectorizer._extract_connected_components(segmentation)

        with open("../data/wcl-1.pickle", 'rb') as f:
            self.wcl = pickle.load(f)
        plot_wcl_against_target(self.wcl, boundary)



    def plot_contours_against_image(self, contours: List[Polygon]):
        from utils import plot_polygon

        plt.imshow(np.array(self.img) + np.nan)
        for c in contours:
            plot_polygon(c, show=False)

        plt.show()

    def plot_cc_against_image(self, ccs: List[SingleConnectedComponent]):
        from utils import plot_binary_image

        bin_arr = np.zeros(self.img.size[::-1], dtype=int)

        for c in ccs:
            bin_arr |= c.array

        plot_binary_image(bin_arr, show=True)

