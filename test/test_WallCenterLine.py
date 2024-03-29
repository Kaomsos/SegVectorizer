import unittest

import pickle
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import segvec.main_steps.wall_center_line
from entity.wall_center_line import WallCenterLine
from segvec.utils import plot_wcl_against_target
from segvec.geometry import find_boundaries, rasterize_polygon, find_connected_components
from segvec.main_steps.room_type import refine_room_types
from segvec.utils import *
from segvec import PaletteConfiguration
from matplotlib.patches import Patch
import json


class TestWallCenterOptimization(unittest.TestCase):
    def test_init_(self):
        # get add_boundary
        path = '../data/flat_0.png'
        self.img = Image.open(path)
        self.boundary = np.zeros(self.img.size[::-1], dtype=int)
        for cc in find_boundaries(self.img):
            self.boundary |= cc.array

        plt.imshow(self.boundary, cmap='gray')
        plt.show()

        path = '../data/countours-1.pickle'
        with open(path, 'rb') as f:
            self.contours = pickle.load(f)

        # plot all contours of rooms
        # plt.imshow(np.array(self.img) + np.nan)
        # for c in self.contours:
        #     plt.plot(c.plot_x, c.plot_y)
        # plt.show()
        print('loaded contours from pickle file')

        # construct a graph from contours
        self.wcl = WallCenterLine(self.contours)
        plot_wcl_against_target(self.wcl,
                                self.boundary + np.nan,
                                title="initial state"
                                )
        pass

    def test_k_neighbor(self):
        from sklearn.neighbors import kneighbors_graph

        self.test_init_()
        g = kneighbors_graph(self.wcl.V, 3, mode='distance', include_self=False)
        g = (g > 0).toarray() & (g < -10).toarray()
        edges = np.argwhere(np.triu(g | g.T)).tolist()
        i2v = self.wcl.i2v
        after_merge = {}
        for i, j in edges:
            i_after = after_merge.get(i, i)
            j_after = after_merge.get(j, j)
            print(f'try merging {i2v[i]} and {i2v[j]}')
            self.wcl.merge_vertices(i2v[i_after], i2v[j_after])
            after_merge[j_after] = i_after
        from segvec.utils import plot_wall_center_lines
        plot_wall_center_lines(self.wcl)
        pass

    def test_undirected_graph(self):
        from segvec.entity.graph import UndirectedGraph
        n_v = 6
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
        g = UndirectedGraph(n_v, edges)
        self.assertTrue(isinstance(g.vertices, list))
        self.assertTrue(isinstance(g.vertices[0], int))
        self.assertTrue(isinstance(g._adjacency_list,  dict))

        g.merge_vertices(0, 3)
        self.assertTrue(0 in g.vertices)
        self.assertTrue(3 not in g.vertices)
        self.assertRaises(AssertionError, g._check_edge_exists, (0, 3))
        self.assertRaises(AssertionError, g._check_edge_exists, (3, 0))
        self.assertTrue(g._edge_exists((0, 2)) and g._edge_exists((0, 4)))

        g.append_vertex_to_edge(0, (1, 2))
        self.assertRaises(AssertionError, g._check_edge_exists, (1, 2))

    def test_segments_matrix(self):
        self.test_init_()
        S, E = self.wcl.segments_matrix
        L = self.wcl.L
        pass

    def test_merge_vertices(self):
        self.test_init_()
        self.interactive_merge_vertices(50, 38)
        self.interactive_append_v_to_e(53, (39, 50))
        self.interactive_append_v_to_e(15, (19, 18))
        self.interactive_merge_vertices(14, 19)

    def test_junction_reduction(self):
        # step1: reduce by condition x
        # step2: reduce by condition y
        import segvec.main_steps.wall_center_line as wclo
        from segvec.utils import plot_wall_center_lines, plot_wcl_against_target

        self.test_init_()

        plot_wcl_against_target(self.wcl, self.boundary, title='before processing')

        reducer = segvec.main_steps.wall_center_line.VertexReducer(self.wcl)

        reducer.reduce_by_condition_x(10)
        plot_wall_center_lines(self.wcl, title=f"reduce_iter=1, after condition x")

        reducer.reduce_by_condition_y(10)
        plot_wall_center_lines(self.wcl, title=f"reduce_iter=1, after condition y")

    def test_coordinate_optimization(self):
        import segvec.main_steps.wall_center_line as wclo
        from segvec.utils import plot_wcl_against_target

        self.test_junction_reduction()
        plot_wcl_against_target(self.wcl, self.boundary, title='after reduction')

        self.optimizer = segvec.main_steps.wall_center_line.CoordinateOptimizer(self.wcl, max_iter=10, lr=0.1)
        self.optimizer.fit(self.boundary, verbose=True)
        plot_wcl_against_target(self.wcl, self.boundary, title='after optimization')

    def test_alternating_optimization(self):
        from segvec.main_steps.wall_center_line import CoordinateOptimizer
        from segvec.main_steps.wall_center_line import VertexReducer
        from segvec.utils import plot_wcl_against_target

        self.test_init_()

        plot_wcl_against_target(self.wcl, self.boundary, title='before processing')

        reducer = VertexReducer(self.wcl)
        optimizer = CoordinateOptimizer(self.wcl, downscale=4, max_iter=10, lr=0.1)
        max_iter = 5

        # iterating for at most max_iter
        for i in range(max_iter):
            reducer._flag = True
            reducer.reduce_by_condition_x(10)
            plot_wcl_against_target(self.wcl, self.boundary + np.nan, title=f'iter={i + 1}, after condition x')

            reducer.reduce_by_condition_y(10)
            plot_wcl_against_target(self.wcl, self.boundary + np.nan, title=f'iter={i + 1}, after condition y')
            if reducer.stop:
                print('meet stopping criterion')
                break
            plot_wcl_against_target(self.wcl, self.boundary, title=f'iter={i + 1}, before optimization')
            optimizer.fit(self.boundary, verbose=False)
            plot_wcl_against_target(self.wcl, self.boundary, title=f'iter={i + 1}, after optimization')
        print("--------------------end--------------------")

        # serialization
        import pickle
        path = '../data/wcl.pickle'
        with open(path, 'wb') as f:
            pickle.dump(self.wcl, f)

        plt.imshow(self.boundary + np.nan)
        plot_rooms_in_wcl(wcl=self.wcl, title="rooms in wall center lines", show=True)

    def interactive_merge_vertices(self, i, j):
        from segvec.utils import plot_wall_center_lines
        self.wcl.merge_vertices(i, j)
        plot_wall_center_lines(self.wcl, title=f"merge vertices {i} and {j} into {i}")

    def interactive_append_v_to_e(self, v, e):
        from segvec.utils import plot_wall_center_lines
        self.wcl.append_vertex_to_edge(v, e)
        plot_wall_center_lines(self.wcl, title=f"append vertex {v} to edge {e}")


class TestWallCenterLine(unittest.TestCase):
    def _init_wcl(self, path='../data/wcl-1.pickle', show=True):
        with open(path, 'rb') as f:
            self.wcl: WallCenterLine = pickle.load(f)

        if show:
            plot_wcl_against_target(self.wcl,
                                    self.boundary + np.nan,
                                    title="optimized wall center lines"
                                    )

            plt.imshow(self.boundary + np.nan)
            plot_rooms_in_wcl(wcl=self.wcl, title="rooms in wall center lines", show=True)

    def _init_img(self, path='../data/flat_0.png', show=True):
        self.img = Image.open(path)
        self.boundary = np.zeros(self.img.size[::-1], dtype=int)
        for cc in find_boundaries(self.img):
            self.boundary |= cc.array
        if show:
            plt.imshow(self.boundary, cmap='gray')
            plt.show()

    def _init_matrix(self, path='../data/seg_reduced.pickle', show=True):
        with open(path, 'rb') as f:
            self.img = pickle.load(f)

        boundaries = []
        for c in [5, 6, 7]:
            boundaries += find_connected_components(self.img, c)

        self.boundary = np.zeros_like(self.img, dtype=int)
        for cc in boundaries:
            self.boundary |= cc.array

        if show:
            plt.imshow(self.img, cmap='tab20', interpolation='none')
            plt.title('segmentation')
            plt.colorbar()
            plt.show()
            plt.imshow(self.boundary, cmap='gray')
            plt.title('boundary')
            plt.show()

    def test_wcl_with_open(self):
        from entity.wall_center_line import WallCenterLineWithOpenPoints
        self._init_wcl()
        self.wcl_o = WallCenterLineWithOpenPoints.from_wcl(self.wcl)
        pass

    def test_room_type(self):
        # get add_boundary
        # wclo_path = '../data/flat_0.png'
        # self._init_img(wclo_path)

        path = '../data/seg_reduced.pickle'
        self._init_matrix(path)

        # construct a graph from contours
        path = '../data/wcl_mpmw.pickle'
        self._init_wcl(path)

        img_shape = self.boundary.shape
        img = np.zeros_like(self.boundary, dtype=int)
        for i, room in enumerate(self.wcl.rooms):
            mask = rasterize_polygon(img_shape, room)
            ras = np.where(mask, i + 1, 0)
            img += ras

        plt.imshow(img, cmap='tab20', interpolation='none')
        plt.colorbar()
        plt.show()

        room_types = refine_room_types(wcl=self.wcl, segmentation=self.img, boundary=(5, 6, 7), background=(0,))
        pass

    def test_wcl(self):
        with open("../data/wcl_out.pickle", 'rb') as f:
            wcl = pickle.load(f)

        with open("../data/wcl_out_seg.pickle", 'rb') as f:
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

        with open("../release/WCL/wall_center_line.json", 'w') as f:
            obj = wcl.json
            json.dump(obj, f, indent=4)

        plot_empty_image_like(seg, show=False)
        plot_wcl_o_against_target(wcl, title='', annotation=True, show=False)
        plot_rooms_in_wcl(wcl, p_config, title="效果示意图", contour=False, show=False)
        handels = [Patch(color='lime'), Patch(color='red')]
        labels = ['门', '窗']
        plt.legend(bbox_to_anchor=(1., 1.), handles=handels, labels=labels)
        plt.show()
#####################################################################
# utils function for plotting


