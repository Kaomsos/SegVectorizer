from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Set
if TYPE_CHECKING:
    from ..typing_ import WallCenterLine

import numpy as np

from segvec.geometry import rasterize_polygon


def count_pixels_in_region(segmentation: np.ndarray,
                           mask: np.ndarray = None
                           ) -> Dict[int, int]:
    """
        return the count of pixels in the region defined by a mask
        if mask is not given, count the pixels in the whole segmentation
    :param segmentation:
    :param mask:
    :return:
    """
    if mask is not None:
        assert segmentation.ndim == mask.ndim
        for x, y in zip(segmentation.shape, mask.shape):
            assert x == y

        segmentation = segmentation[mask]
    else:
        segmentation = segmentation.flatten()

    uniq, cnt = np.unique(segmentation, return_counts=True)
    return dict(zip(uniq.tolist(), cnt.tolist()))


def is_trivial(counts: Dict[int, int], threshold=10) -> bool:
    return sum(counts.values()) < threshold


def is_non_room(counts: Dict, threshold=0.5, non_room: Set[int] = ()) -> bool:
    total = sum(counts.values())
    non_room_cnt = sum(item[1] for item in filter(lambda item: item[0] in non_room, counts.items()))
    return non_room_cnt / total > threshold


def find_maximum_room_type(counts: Dict[int, int], non_room: Set[int] = ()) -> int:
    room_cnts = filter(lambda item: item[0] not in non_room, counts.items())
    maximum = max(room_cnts, key=lambda item: item[1])
    return maximum[0]


def refine_room_types(wcl: WallCenterLine,
                      segmentation: np.ndarray,
                      boundary: Set[int],
                      background: Set[int],
                      trivial_threshold=10,
                      non_room_threshold=0.5
                      ):
    boundary = set(boundary)
    background = set(background)
    room_types = []
    for room in wcl.rooms:
        mask = rasterize_polygon(arr_shape=segmentation.shape, polygon=room)
        counts = count_pixels_in_region(segmentation, mask)
        if is_trivial(counts, trivial_threshold):
            room_types.append(None)
        elif is_non_room(counts, non_room_threshold, non_room=boundary | background):
            room_types.append(None)
        else:
            room_types.append(find_maximum_room_type(counts, non_room=boundary | background))

    return room_types
