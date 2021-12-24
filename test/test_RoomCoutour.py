import unittest

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


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
