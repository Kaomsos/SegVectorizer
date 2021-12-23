import importlib

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import unittest


class TestFindConnectedComponents(unittest.TestCase):
    def test_init(self):
        path = 'data/flat_1.png'
        self.img = Image.open(path)
        self.palette = self.img.getcolors()

    def test_find_components(self):
        from geometry import find_connected_components, palette
        self.test_init()
        for color in palette.values():
            c = find_connected_components(self.img, color, 0)
            print(f'n_connected_components = {len(c)} for color: {color}')

    def test_find_rooms(self):
        from geometry import find_rooms
        self.test_init()
        rooms = find_rooms(self.img)

        plt.imshow(list(rooms.values())[0][0].array + np.nan)
        for ccs in rooms.values():
            for cc in ccs:
                plt.plot(cc.boundary[..., 0], cc.boundary[..., 1])
        plt.show()

    def test_SingleCC(self):
        from entity_class import SingleConnectedComponent
        from geometry import find_connected_components, palette, find_boundaries
        from utils import plot_binary_image
        self.test_init()

        all_ = find_boundaries(self.img, threshold=5)
        # for type_ in ['bathroom/washroom', 'livingroom/kitchen/dining room',
        #           'bedroom', 'hall', 'balcony', 'closet']:
        #     ccs = find_connected_components(self.img, palette[type_], threshold=0)
        #     print(f'n_connected_components = {len(ccs)} for type: {type_}')
        #     all_ += ccs

        img = np.zeros((self.img.size[1], self.img.size[0]), dtype=int)
        for cc in all_:
            img |= cc.array

        plot_binary_image(img)


class TestRoomContourOptimization(unittest.TestCase):
    def test_init(self):
        from geometry import find_rooms
        path = path = 'data/flat_0.png'
        self.img = Image.open(path)
        rooms = find_rooms(self.img)

        self.rooms = []
        for ccs in rooms.values():
            for cc in ccs:
                self.rooms.append(cc)

        self.assertTrue(isinstance(self.rooms, list))
        print(f'{len(self.rooms)} rooms are found!')
        # self.plot_rooms()

    def test_vertices_reduction(self):
        from image_reader import BinaryImageFromFile
        from entity_class import Polygon
        from room_contour_optimization import VertexReducer
        from utils import plot_polygon

        path = "data/rect2.jpg"
        img = BinaryImageFromFile(path)
        plg = Polygon(arr=img.array, tol=0)
        plot_polygon(img, plg)
        for d in [0, 20, 30, 40, 50, 60]:
            reducer = VertexReducer(plg=plg, delta_a=0, delta_b=d)
            reducer.reduce()
            plot_polygon(img, plg)
        pass

    def test_coordinate_optimization(self):
        from image_reader import BinaryImageFromFile
        import torch
        from torch.optim import RMSprop
        from objective import log_iou, boundary, orthogonal
        from rasterizer import Base2DPolygonRasterizer

        path = "data/rect1.jpg"
        self.sigma = 10
        rounds = 10
        rect_img = BinaryImageFromFile(path, scale=0.1)
        rasterizer = Base2DPolygonRasterizer(image_size=rect_img.size,
                                             mode="soft euclidean",
                                             sigma=self.sigma)
        bbox = rect_img.bbox
        plg = torch.tensor([(bbox[0][0], bbox[0][1]),
                            (bbox[0][0], bbox[1][1]),
                            ((bbox[0][0] + bbox[1][0]) / 2, bbox[1][1] + 30),
                            (bbox[1][0], bbox[1][1]),
                            (bbox[1][0], bbox[0][1])], dtype=torch.float64, requires_grad=True)
        # pass a iterable or dictionary of tensors
        optimizer = RMSprop([plg], lr=0.5)
        for i in range(rounds):
            self.i = i
            rendered = rasterizer(plg)
            iou: torch.Tensor = log_iou(rendered, rect_img.array)
            bo: torch.Tensor = boundary(plg, rect_img.array)
            orth: torch.Tensor = orthogonal(plg)
            loss = 1 * bo + 1 * orth - 5 * iou

            self.plot_soft_ras(rendered, rect_img)

            loss.backward()
            optimizer.step()
            print(f"iter = {i + 1}, gradient = {plg.grad}, -log_iou = {-iou.detach().numpy()}")

    def test_alternating_optimization(self):
        from utils import plot_polygon, plot_contours
        from entity_class import Polygon
        self.test_init()
        opt_contours = []
        for i, cc in enumerate(self.rooms):
            # if i != 12:
            #     continue
            plg = self.alternating_optimizing(cc)
            # plg = alternating_optimize(cc)
            self.assertTrue(isinstance(plg, Polygon))
            opt_contours.append(plg)
            print(f'the {i+1}th room has been optimized')
        ######################################
        # pickle the contours
        # pickle = importlib.import_module('pickle')
        # path = 'data/countours-1.pickle'
        # with open(path, 'wb') as f:
        #     pickle.dump(opt_contours, f)
        print('--- end ---')

    @staticmethod
    def alternating_optimizing(cc):
        from utils import plot_polygon, plot_contours
        from entity_class import Polygon
        from room_contour_optimization import VertexReducer
        from room_contour_optimization import CoordinateOptimizer

        plot_contours(cc)
        plg = Polygon(connected_component=cc, tol=2)
        reducer = VertexReducer(plg, delta_a=10)
        opt = CoordinateOptimizer(target=cc, max_iter=0, sigma=10)
        print(f'num_vertices = {len(plg)}')
        for _ in range(5):
            plot_polygon(cc, plg)
            reducer.reduce()
            plot_polygon(cc, plg)
            print(f'num_vertices = {len(plg)}')
            if reducer.stop:
                print('--- meet stopping criterion! ---')
                break
            opt.fit(plg, verbose=True)
        del opt
        del reducer
        return plg

    def plot_rooms(self):
        plt.imshow(self.rooms[0].array + np.nan)
        for cc in self.rooms:
            plt.plot(cc.boundary[..., 0], cc.boundary[..., 1])
        plt.show()


class TestWallCenterOptimization(unittest.TestCase):
    def test_init_(self):
        import pickle
        from entity_class import WallCenterLine
        from utils import plot_wall_center_lines, plot_wcl_against_target
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
        from utils import plot_wcl_against_target, plot_wall_center_lines

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


class TestOpenPointExtraction(unittest.TestCase):
    def _init(self, path='data/flat_1.png'):
        self.img = Image.open(path)
        self.palette = self.img.getcolors()

    def test_find_door_window(self):
        from geometry import find_connected_components, palette
        from utils import plot_binary_image

        self._init(path='data/flat_1.png')

        color = palette["door&window"]
        self.dws = find_connected_components(self.img, color, threshold=5)

        self.all_cc = np.zeros_like(self.dws[0].array, dtype=int)
        for i, dw in enumerate(self.dws):
            self.all_cc |= dw.array
            plot_binary_image(dw.array, title=f"component {i + 1}")
            # assert bbox attribute
            self.assertTrue(isinstance(dw.bbox, tuple) and len(dw.bbox) == 2)
            # assert boundary attribute
            self.assertTrue(isinstance(dw.boundary, np.ndarray)
                            and len(dw.boundary.shape) == 2
                            and dw.boundary.shape[1] == 2)

        # plot everything
        plot_binary_image(self.all_cc, "all components")

    def test_rect_fit(self):
        from geometry import find_connected_components, palette
        from room_contour_optimization import RectangleOptimizer
        from utils import plot_binary_image

        # get all door/window components
        self._init(path='data/flat_0.png')

        color = palette["door&window"]
        self.dws = find_connected_components(self.img, color, threshold=5)

        rect_fitter = RectangleOptimizer(sigma=0.1,
                                         max_iter=10,
                                         lr=0.01,
                                         log_iou_target=-0.5)

        l = []
        for dw in self.dws[:]:
            rect = rect_fitter.fit_return(dw, verbose=True)
            l.append(rect)
            print(rect)

        # import pickle
        # path = 'data/door_window.pickle'
        # with open(path, 'wb') as f:
        #     pickle.dump(l, f)

    def test_rect_process(self):
        # init and get img
        # self._init()

        # init and get all doors and windows
        self.test_find_door_window()

        # load doors and windows
        import pickle
        path = 'data/door_window.pickle'
        with open(path, 'rb') as f:
            l = pickle.load(f)

        def get_segments_ends(rect):
            center = np.array(rect[0])
            w, h, theta = rect[1]
            R = np.array([[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta), np.cos(theta)]])
            if w < h:
                length = h
                ends = center + R @ np.array([[0, h / 2],
                                              [0, - h / 2]])
            else:
                length = w
                ends = center + R @ np.array([[w / 2, 0],
                                              [- w / 2, 0]])
            return ends

        def plot_position_of_rects(l):
            for i, rect in enumerate(l):
                center = rect[0]
                ends = get_segments_ends(rect)
                plt.plot(ends[..., 0], ends[..., 1], color="#3399ff")
                plt.text(center[0]+5, center[1]-5, f"{i}", size='x-small')

        # plot doors and windows
        plt.imshow(np.array(self.img) + np.nan)
        plot_position_of_rects(l)
        plt.title("positions of doors & windows")
        plt.show()

        # plot doors and windows against connected componets
        plt.imshow(1 - self.all_cc, cmap='gray')
        plot_position_of_rects(l)
        plt.title("positions of doors & windows against target")
        plt.show()

        # save ends
        # path = 'data/door_window_ends.pickle'
        # l = [get_segments_ends(rect) for rect in l]
        # with open(path, 'wb') as f:
        #     pickle.dump(l, f)

    def test_plot_rooms_contours(self):
        import pickle
        from geometry import find_rooms

        self._init('data/flat_0.png')

        arr = np.zeros(np.array(self.img).shape[:2], dtype=int)
        for rooms in find_rooms(self.img).values():
            for cc in rooms:
                arr |= cc.array
        plt.imshow(arr, cmap='gray')
        plt.title("all rooms")
        plt.show()

        path = 'data/countours-1.pickle'
        with open(path, 'rb') as f:
            self.contours = pickle.load(f)

        # plot all contours of rooms
        plt.imshow(np.array(self.img) + np.nan)
        for c in self.contours:
            plt.plot(c.plot_x, c.plot_y)
        plt.title("all rooms contours")
        plt.show()

        # plot all contours against rooms pixles
        plt.imshow(1 - arr, cmap='gray')
        for c in self.contours:
            plt.plot(c.plot_x, c.plot_y)
        plt.title("all rooms contours")
        plt.show()

    def test_plot_wcl(self):
        from utils import plot_wall_center_lines

        self._init()
        self._load_wcl()

        arr = np.zeros(np.array(self.img).shape[:2])
        plt.imshow(arr + np.nan)
        plot_wall_center_lines(self.wcl)

    def _load_wcl(self, path='data/wcl.pickle'):
        import pickle
        with open(path, 'rb') as f:
            self.wcl = pickle.load(f)

    def _load_door_windows(self, path='data/door_window_ends.pickle'):
        import pickle
        with open(path, 'rb') as f:
            self.open_ends = pickle.load(f)

    def test_anchor_open_ends(self):
        from geometry import distance_seg_to_segments, project_seg_to_seg
        from utils import plot_wall_center_lines

        self._init()
        self._load_wcl()
        self._load_door_windows()

        wcl = self.wcl
        segments_collections = self.wcl.segments_collection
        open_ends = self.open_ends

        plt.imshow(np.zeros(np.array(self.img).shape[:2]) + np.nan)
        plot_wall_center_lines(wcl, show=False)
        for ends in open_ends:
            plt.plot(ends[..., 0], ends[..., 1], color="#da420f")
        plt.title("before projection")
        plt.show()

        projections = []
        for ends in open_ends:
            ds = distance_seg_to_segments(ends, segments_collections)
            which = ds.argmin()
            to = (segments_collections[0][which], segments_collections[1][which])
            proj = project_seg_to_seg(ends, to)
            projections.append(proj)

        plt.imshow(np.zeros(np.array(self.img).shape[:2]) + np.nan)
        plot_wall_center_lines(wcl, show=False)
        for proj in projections:
            plt.plot(proj[..., 0], proj[..., 1], color="#da420f")
        plt.title("after projection")
        plt.show()


class TestPCA(unittest.TestCase):
    def _init(self, path='data/flat_1.png'):
        self.img = Image.open(path)
        self.palette = self.img.getcolors()

    def test_find_door_window(self):
        from geometry import find_connected_components, palette
        from utils import plot_binary_image

        self._init(path='data/flat_1.png')

        color = palette["door&window"]
        self.dws = find_connected_components(self.img, color, threshold=5)

        self.all_cc = np.zeros_like(self.dws[0].array, dtype=int)
        for i, dw in enumerate(self.dws):
            self.all_cc |= dw.array

            # plot this current connected components
            # plot_binary_image(dw.array, title=f"component {i + 1}")

            # assert bbox attribute
            self.assertTrue(isinstance(dw.bbox, tuple) and len(dw.bbox) == 2)
            # assert boundary attribute
            self.assertTrue(isinstance(dw.boundary, np.ndarray)
                            and len(dw.boundary.shape) == 2
                            and dw.boundary.shape[1] == 2)

        # plot everything
        plot_binary_image(self.all_cc, "all components")

    def test_PCA(self):
        from geometry import find_connected_components, palette
        from entity_class import SingleConnectedComponent
        from sklearn.decomposition import PCA

        self._init(path='data/flat_1.png')

        color = palette["door&window"]
        self.dws = find_connected_components(self.img, color, threshold=5)

        def fit_one(dw: SingleConnectedComponent):
            X = np.argwhere(dw.array)[..., ::-1]
            pca = PCA(n_components=2)
            trans_X = pca.fit_transform(X=X)[..., :: -1]
            return trans_X

        for i, dw in enumerate(self.dws):
            fitted = fit_one(dw)
            pass





if __name__ == '__main__':
    case = TestWallCenterOptimization()
    case.test_coordinate_optimization()
