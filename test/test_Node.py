import unittest
from entity.graph import Node, NodeWith2DCoordinate, EdgeMy


class TestNode(unittest.TestCase):
    def _init(self):
        pass

    def test_hash(self):

        class RandomObj:
            def __init__(self):
                self.a = (1, 2, 3)

        o1 = RandomObj()
        o2 = RandomObj()
        print(f"hash(o1) = {hash(o1)}, hash(o2) = {hash(o2)}")
        print(f"id(o1) = {id(o1)}, id(o2) = {id(o2)}")
        print(f"assert o1 == o2, {o1 == o2}")

    def test_node_init(self):
        node1 = Node()
        node2 = Node(weight=13, config={}, sdf="sfsasae")
        self.assertTrue(hasattr(node2, "weight"))
        self.assertTrue(hasattr(node2, "config"))
        self.assertTrue(hasattr(node2, "sdf"))

    def test_edge_init(self):
        node1 = Node()
        node2 = Node(weight=13, config={}, sdf="sfsasae")
        e1 = EdgeMy((node1, node2))
        self.assertTrue(isinstance(e1, tuple))
        self.assertTrue(isinstance(e1[0], Node) and e1[0] is node1)
        self.assertTrue(isinstance(e1[1], Node) and e1[1] is node2)
