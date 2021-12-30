from __future__ import annotations
from typing import TypeVar, TYPE_CHECKING
import numpy as np
import numpy.typing as npt

import unittest


class TestTyping(unittest.TestCase):
    def test_numpy_typing(self):
        T1 = TypeVar("T1", bound=npt.NBitBase)
        T2 = TypeVar("T2", bound=npt.NBitBase)

        def add(a: np.floating[T1], b: np.integer[T2]) -> np.floating[T1 | T2]:
            return a + b

        a = np.float16()
        b = np.int64()
        out = add(a, b)

        if TYPE_CHECKING:
            reveal_locals()
            # note: Revealed local types are:
            # note:     a: numpy.floating[numpy.typing._16Bit*]
            # note:     b: numpy.signedinteger[numpy.typing._64Bit*]
            # note:     out: numpy.floating[numpy.typing._64Bit*]
