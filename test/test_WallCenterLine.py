import unittest

import pickle
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import segvec.main_steps.wall_center_line
from entity.wall_center_line import WallCenterLine
from segvec.utils import plot_wcl_against_target
from segvec.geometry import find_boundaries, rasterize_polygon, find_connected_components, get_bounding_box
from segvec.main_steps.room_type import count_pixels_in_region, refine_room_types


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
        # path = '../data/flat_0.png'
        # self._init_img(path)

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


#####################################################################
# utils function for plotting
def plot_rooms_in_wcl(wcl: WallCenterLine,
                      palette: dict = None,
                      title: str = "",
                      show: bool = False
                      ):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    reverse_palette = {v: k for k, v in palette.items()}
    for room, type_ in zip(wcl.rooms, wcl.room_types):
        indices = list(range(room.shape[0])) + [0]
        p1, p2 = get_bounding_box(room)
        x, y = (p1 + p2) / 2

        if palette is None:
            text = f'type {type_}'
        else:
            text = reverse_palette[type_]

        plt.plot(room[indices, 0], room[indices, 1])
        plt.text(x, y, text, size='x-small', ha='center', va='center')
    plt.title(title)
    if show:
        plt.show()


