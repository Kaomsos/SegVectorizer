from __future__ import annotations
from typing import List, TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from segvec.typing_ import SingleConnectedComponent, Polygon
import unittest
from PIL import Image
import pickle
import numpy as np

from segvec import convert_a_segmentation, PaletteConfiguration
from test_WallCenterLine import plot_rooms_in_wcl


class TestVectorizer(unittest.TestCase):
    def _init(self, path='../data/flat_1.png'):
        from segvec.vetorizer import Vectorizer, PaletteConfiguration
        from segvec.utils import palette

        self.img = Image.open(path)
        self.segmentation = np.array(self.img)

        p_config = PaletteConfiguration(palette)
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
            self.assertTrue(isinstance(e, tuple))
        self.assertTrue(isinstance(p_config.opens, set))
        for e in p_config.opens:
            self.assertTrue(isinstance(e, tuple))
        self.assertTrue(isinstance(p_config.boundaries, set))
        for e in p_config.boundaries:
            self.assertTrue(isinstance(e, tuple))

        self.vectorizer = Vectorizer(palette_config=p_config)

    def test_vectorizer_get_components(self):
        from segvec.entity.image import SingleConnectedComponent

        self._init(path='../data/flat_1.png')

        segmentation = np.array(self.img)

        c1, c2, c3 = self.vectorizer._extract_connected_components(segmentation)
        self.assertTrue(isinstance(c2, np.ndarray))
        for ccs in [c1, c3]:
            self.assertTrue(isinstance(ccs, list))
            for cc in ccs:
                self.assertTrue(isinstance(cc, SingleConnectedComponent))

    def test_vectorize_get_wall_center_lines(self):
        from segvec.typing_ import Contour, WallCenterLine

        self._init(path='../data/flat_0.png')

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
        from segvec.utils import plot_wcl_against_target, plot_position_of_rects, plot_wcl_o_against_target
        from segvec import Vectorizer, PaletteConfiguration
        from segvec.utils import palette

        path = '../data/Figure_47541863.png'
        self.img = Image.open(path)
        self.segmentation = np.array(self.img)

        p_config = PaletteConfiguration(palette)
        p_config.add_open("door&window")
        for item in ['bathroom/washroom',
                     'livingroom/kitchen/dining add_room',
                     'bedroom',
                     'hall',
                     'balcony',
                     'closet']:
            p_config.add_room(item)
        p_config.add_boundary("wall")

        self.vectorizer = Vectorizer(palette_config=p_config)
        wcl_o = self.vectorizer._vectorize(self.segmentation)
        plot_wcl_o_against_target(wcl_o, self.vectorizer.boundary)

    def plot_contours_against_image(self, contours: List[Polygon]):
        from segvec.utils import plot_polygon

        plt.imshow(np.array(self.img) + np.nan)
        for c in contours:
            plot_polygon(c, show=False)

        plt.show()

    def plot_cc_against_image(self, ccs: List[SingleConnectedComponent]):
        from segvec.utils import plot_binary_image

        bin_arr = np.zeros(self.img.size[::-1], dtype=int)

        for c in ccs:
            bin_arr |= c.array

        plot_binary_image(bin_arr, show=True)

    def test_2d_array(self):
        with open("../data/seg_reduced.pickle", 'rb') as f:
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

        plt.imshow(seg, cmap='tab20', interpolation='none')
        plt.show()

        wcl = convert_a_segmentation(seg, p_config)

        plt.imshow(seg + np.nan)
        plot_rooms_in_wcl(wcl, show=True)

        path = '../data/wcl_mpmw.pickle'
        with open(path, 'wb') as f:
            pickle.dump(wcl, f)
