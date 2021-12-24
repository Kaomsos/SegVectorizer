import unittest

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


class TestWallCenterOptimization(unittest.TestCase):
    def test_init_(self):
        import pickle
        from entity_class import WallCenterLine
        from utils import plot_wcl_against_target
        from geometry import find_boundaries

        # get boundary
        path = path = 'data/flat_0.png'
        self.img = Image.open(path)
        self.boundary = np.zeros(self.img.size[::-1], dtype=int)
        for cc in find_boundaries(self.img):
            self.boundary |= cc.array

        plt.imshow(self.boundary, cmap='gray')
        plt.show()

        path = 'data/countours-1.pickle'
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
        plot_wcl_against_target(self.boundary + np.nan, self.wcl, title="initial state")

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
        from utils import plot_wall_center_lines
        plot_wall_center_lines(self.wcl)
        pass

    def test_undirected_graph(self):
        from entity_class import UndirectedGraph
        n_v = 6
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
        g = UndirectedGraph(n_v, edges)
        self.assertTrue(isinstance(g.vertices, list))
        self.assertTrue(isinstance(g.vertices[0], int))
        self.assertTrue(isinstance(g._adjacency_list,  dict))
        self.assertTrue(isinstance(g.matrix, np.ndarray) and g.matrix.shape[0] == g.matrix.shape[1] == n_v)

        g.merge_vertices(0, 3)
        self.assertTrue(0 in g.vertices)
        self.assertTrue(3 not in g.vertices)
        self.assertRaises(AssertionError, g._check_edge_exists, (0, 3))
        self.assertRaises(AssertionError, g._check_edge_exists, (3, 0))
        self.assertTrue(g._check_edge_exists((0, 2)) and g._check_edge_exists((0, 4)))
        self.assertTrue(g.matrix.shape[0] == g.matrix.shape[1] == n_v - 1)

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
        import wall_centerline_optimization as wclo
        from utils import plot_wall_center_lines, plot_wcl_against_target

        self.test_init_()

        plot_wcl_against_target(self.boundary, self.wcl, title='before processing')

        reducer = wclo.VertexReducer(self.wcl)

        reducer.reduce_by_condition_x(10)
        plot_wall_center_lines(self.wcl, title=f"reduce_iter=1, after condition x")

        reducer.reduce_by_condition_y(10)
        plot_wall_center_lines(self.wcl, title=f"reduce_iter=1, after condition y")

    def test_coordinate_optimization(self):
        import wall_centerline_optimization as wclo
        from utils import plot_wcl_against_target

        self.test_junction_reduction()
        plot_wcl_against_target(self.boundary, self.wcl, title='after reduction')

        self.optimizer = wclo.CoordinateOptimizer(self.wcl, max_iter=10, lr=0.1)
        self.optimizer.fit(self.boundary, verbose=True)
        plot_wcl_against_target(self.boundary, self.wcl, title='after optimization')

    def test_alternating_optimization(self):
        from wall_centerline_optimization import VertexReducer, CoordinateOptimizer
        from utils import plot_wcl_against_target

        self.test_init_()

        plot_wcl_against_target(self.boundary, self.wcl, title='before processing')

        reducer = VertexReducer(self.wcl)
        optimizer = CoordinateOptimizer(self.wcl, downscale=4, max_iter=10, lr=0.1)
        max_iter = 5

        # iterating for at most max_iter
        for i in range(max_iter):
            reducer._flag = True
            reducer.reduce_by_condition_x(10)
            plot_wcl_against_target(self.boundary + np.nan, self.wcl, title=f'iter={i + 1}, after condition x')

            reducer.reduce_by_condition_y(10)
            plot_wcl_against_target(self.boundary + np.nan, self.wcl, title=f'iter={i + 1}, after condition y')
            if reducer.stop:
                print('meet stopping criterion')
                break
            plot_wcl_against_target(self.boundary, self.wcl, title=f'iter={i + 1}, before optimization')
            optimizer.fit(self.boundary, verbose=False)
            plot_wcl_against_target(self.boundary, self.wcl, title=f'iter={i + 1}, after optimization')
        print("--------------------end--------------------")

        # serialization
        # import pickle
        # path = 'data/wcl.pickle'
        # with open(path, 'wb') as f:
        #     pickle.dump(self.wcl, f)

    def interactive_merge_vertices(self, i, j):
        from utils import plot_wall_center_lines
        self.wcl.merge_vertices(i, j)
        plot_wall_center_lines(self.wcl, title=f"merge vertices {i} and {j} into {i}")

    def interactive_append_v_to_e(self, v, e):
        from utils import plot_wall_center_lines
        self.wcl.append_vertex_to_edge(v, e)
        plot_wall_center_lines(self.wcl, title=f"append vertex {v} to edge {e}")

