from unittest import TestCase
from segvec.entity.mapping import SemiIdentityMapping


class TestSemiIdentityMapping(TestCase):
    def test_identity(self):
        mapping = SemiIdentityMapping()
        list_ = [1, 4, 3534, (2, 1, 3), "fsd"]
        for e in list_:
            self.assertTrue(e == mapping[e])
            self.assertTrue(e == mapping(e))

    def test_semi(self):
        mapping = SemiIdentityMapping()
        mapping.update({
            1: 1,
            2: 1,
            3: 2,
        })

        for e in [1, 2, 3]:
            self.assertTrue(1 == mapping[e])
            self.assertTrue(1 == mapping(e))

        for e in [1, 4, 3534, (2, 1, 3), "fsd"]:
            self.assertTrue(e == mapping[e])
            self.assertTrue(e == mapping(e))


