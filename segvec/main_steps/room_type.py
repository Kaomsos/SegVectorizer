from __future__ import annotations
from typing import TYPE_CHECKING, Dict
if TYPE_CHECKING:
    from ..typing_ import *

import numpy as np


def count_pixels_in_region(segmentation: np.ndarray,
                           mask: np.ndarray = None
                           ) -> Dict[int, int]:
    if mask is not None:
        assert segmentation.ndim == mask.ndim
        for x, y in zip(segmentation.shape, mask.shape):
            assert x == y

        segmentation = segmentation[mask]
    else:
        segmentation = segmentation.flatten()

    uniq, cnt = np.unique(segmentation, return_counts=True)
    return dict(zip(uniq.tolist(), cnt.tolist()))


