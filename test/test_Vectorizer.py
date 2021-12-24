import unittest

from PIL import Image


class TestVectorizer(unittest.TestCase):
    def _init(self, path='data/flat_1.png'):
        self.img = Image.open(path)
        self.palette = self.img.getcolors()

    def test_vectorizer_get_components(self):
        pass
