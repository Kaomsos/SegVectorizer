from __future__ import annotations
from typing import TYPE_CHECKING, Dict
if TYPE_CHECKING:
    from ..typing_ import *

import numpy as np


def count_pixels_in_region(segmentation: np.ndarray,
                           mask: np.ndarray = None
                           ) -> Dict[int, int]:
    if mask is not None:
        segmentation = segmentation[mask]

    pass

