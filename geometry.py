from __future__ import annotations
import torch
import numpy as np
from torch import Tensor
from typing import Tuple, List, Dict
from typing_ import SegmentCollection
from skimage.measure import label
from entity_class import SingleConnectedComponent

EPSILON = 1e-2


def get_segments(polygon: Tensor) -> Tuple[Tensor, Tensor]:
    # roll and subtract
    start_p = torch.as_tensor(polygon)
    end_p = torch.roll(polygon, shifts=1, dims=0)
    return start_p, end_p


def distance_p_to_plg(point: Tensor | Tuple[int, int], polygon: Tensor, square=True):
    point = torch.as_tensor(point)
    ds = []
    for s_p, e_p in zip(*get_segments(polygon)):
        d_square = distance_p_to_segment(point, torch.stack([s_p, e_p]), square=True)
        ds.append(d_square)
    if square:
        return torch.stack(ds).min()
    else:
        return torch.sqrt(torch.stack(ds).min() + EPSILON)


def distance_p_to_segment(point: Tensor,
                          segment: Tensor,
                          square=True
                          ) -> torch.Tensor:
    uv = segment - point
    u = uv[0]
    v = uv[1]
    u_sub_v = u - v

    # the following variables don't require gradients
    with torch.no_grad():
        norm_u_sub_v_ = (u_sub_v * u_sub_v).sum()
        proj_u_ = (u * u_sub_v).sum()

    if proj_u_ <= 0:
        d_square = (u * u).sum()
    elif proj_u_ > norm_u_sub_v_:
        d_square = (v * v).sum()
    else:
        d_square = torch.square(u[0] * v[1] - v[0] * u[1]) / (u_sub_v * u_sub_v).sum()

    # square or non-square
    if square:
        return d_square
    else:
        return torch.sqrt(d_square + EPSILON)


def distance_p_to_segments_tensor(point: np.ndarray | torch.Tensor,
                                  segments: Tuple[torch.Tensor, torch.Tensor]
                                  ) -> torch.Tensor:
    point = torch.as_tensor(point)
    ds = []
    for sp, ep in zip(*segments):
        seg = torch.stack([sp, ep])
        distance = distance_p_to_segment(point, seg, square=False)
        ds.append(distance)
    return min(ds)


def distance_p_to_segments(point: np.ndarray, segments: SegmentCollection) -> np.ndarray:
    """
        brutal-force computation
        :param point:
        :param segments:
        :return:
    """
    us = segments[0] - point
    vs = segments[1] - point
    u_sub_v = us - vs
    n_seg = u_sub_v.shape[0]

    square_seglen = (u_sub_v * u_sub_v).sum(axis=-1)

    proj_u = (us * u_sub_v).sum(axis=-1)

    ds = np.zeros(n_seg)
    # case 1
    bool_map_1 = (proj_u <= 0)
    ds[bool_map_1] = np.sqrt((us[bool_map_1] * us[bool_map_1]).sum(axis=-1))
    # case 2
    bool_map_2 = (proj_u > square_seglen)
    ds[bool_map_2] = np.sqrt((vs[bool_map_2] * vs[bool_map_2]).sum(axis=-1))
    # case 3
    bool_map_3 = ~(bool_map_1 | bool_map_2)
    ds[bool_map_3] = np.abs(
                        us[bool_map_3][..., 0] * vs[bool_map_3][..., 1]
                        - us[bool_map_3][..., 1] * vs[bool_map_3][..., 0]
                        ) \
                      / np.sqrt(square_seglen[bool_map_3])

    return ds


def distance_seg_to_segments(seg: Tuple[np.ndarray, np.ndarray], segments: SegmentCollection):
    p1, p2 = seg
    d1 = distance_p_to_segments(p1, segments)
    d2 = distance_p_to_segments(p2, segments)
    d = d1 + d2
    return d


# TODO: add condition on "overflow"
def project_seg_to_seg(src: Tuple[np.ndarray, np.ndarray],
                       to: Tuple[np.ndarray, np.ndarray]
                       ) -> np.ndarray:
    """
        a simple prototype of line projection instead of segment projection
    :param src:
    :param to:
    :return:
    """
    p1, p2 = src
    proj = []
    for p in [p1, p2]:
        A = (to[0] - to[1])[1]
        B = -(to[0] - to[1])[0]
        C = to[0][0] * to[1][1] - to[0][1] * to[1][0]

        x = (B * B * p[0] - A * B * p[1] - A * C) / (A * A + B * B)
        y = (- A * B * p[0] + A * A * p[1] - B * C) / (A * A + B * B)
        foot = np.array([x, y])
        proj.append(foot)

    return np.array(proj)


def intersection_given_x_plg(x, polygon):
    start_p, end_p = get_segments(polygon)
    ys = []
    for s, e in zip(start_p, end_p):
        y = None
        if (s[0] >= x >= e[0]) or (s[0] <= x <= e[0]):
            if torch.absolute(s[0] - e[0]) > 0:
                y = (torch.absolute(s[0] - x) * e[1] + torch.absolute(e[0] - x) * s[1]) / torch.absolute(s[0] - e[0])
        ys.append(y)
    return sorted(filter(lambda v: v is not None, ys), reverse=True)


def get_bounding_box(points: Tensor) -> Tuple[Tensor, Tensor]:
    p1 = torch.amin(points, dim=0)
    p2 = torch.amax(points, dim=0)
    return p1, p2


def find_connected_components(img,
                              color: Tuple[int, int, int],
                              threshold: float = 5
                              ) -> List[SingleConnectedComponent]:
    arr = np.array(img)[..., :3]
    assert len(arr.shape) == 3
    assert arr.shape[-1] == 3

    iter_ = zip(np.moveaxis(arr, -1, 0), color)
    bool_map = np.stack([channel == c for channel, c in iter_], axis=-1)
    bin_arr = np.where(bool_map.all(axis=-1), 1, 0)

    # the core function (skimage.measure.label)
    labels, num = label(bin_arr, return_num=True, background=0)

    connected_components = []
    for i in range(num):
        i += 1
        bin_array = np.where(labels == i, 1, 0)
        inst = SingleConnectedComponent(bin_array, threshold=threshold)
        if inst is not None:
            connected_components.append(inst)

    return connected_components


def find_rooms(img, threshold=10) -> Dict[str, List[SingleConnectedComponent]]:
    """
    :reference
        palette
    :param
        img:
    :return:
    """
    rooms = {}
    for type_ in ['bathroom/washroom', 'livingroom/kitchen/dining room',
                  'bedroom', 'hall', 'balcony', 'closet']:
        cc = find_connected_components(img, palette[type_], threshold=threshold)
        rooms[type_] = cc
    return rooms


def find_boundaries(img, threshold=0) -> List[SingleConnectedComponent]:
    """
        :reference
            palette
        :param
            img:
        :return:
    """
    boundaries = []
    for type_ in ['wall', 'door&window']:
        cc = find_connected_components(img, palette[type_], threshold=threshold)
        boundaries += cc
    return boundaries



palette = {
     'background': [255, 255, 255],
     'closet': [192, 192, 224],
     'bathroom/washroom': [192, 255, 255],
     'livingroom/kitchen/dining room': [224, 255, 192],
     'bedroom': [255, 224, 128],
     'hall': [255, 160, 96],
     'balcony': [255, 224, 224],
     7: [224, 224, 224],  # not used
     8: [224, 224, 128],  # not used
     'door&window': [255, 60, 128],
     'wall': [0, 0, 0]
}
