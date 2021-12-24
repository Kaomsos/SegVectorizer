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
        from typing_ import SingleConnectedComponent

        self._init(path='../data/flat_1.png')

        segmentation = np.array(self.img)

        c1, c2, c3 = self.vectorizer._extract_connected_components(segmentation)
        for ccs in [c1, c2, c3]:
            self.assertTrue(isinstance(ccs, list))
            for cc in ccs:
                self.assertTrue(isinstance(cc, SingleConnectedComponent))



