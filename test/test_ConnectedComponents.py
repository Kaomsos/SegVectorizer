import unittest

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


class TestFindConnectedComponents(unittest.TestCase):
    def test_init(self):
        path = '../data/flat_1.png'
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
        from geometry import find_boundaries
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

