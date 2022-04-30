import unittest
from segvec.entity.wall_center_line import WallCenterLine, WallCenterLineWithOpenPoints
import pickle


class TestSerialization(unittest.TestCase):
    def test_json(self):
        wcl_o = load_wcl_o('../experiments/raw_demo/2_1k8_wcl.pickle')
        self.assertIsInstance(wcl_o, WallCenterLineWithOpenPoints)
        self.assertIsInstance(wcl_o.json, dict)


def load_wcl_o(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj