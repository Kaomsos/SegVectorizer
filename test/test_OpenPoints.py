import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import unittest

from test.test_WallCenterLine import TestWallCenterOptimization


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
