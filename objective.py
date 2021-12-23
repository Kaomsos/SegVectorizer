from __future__ import annotations
from typing import Tuple
import torch
import numpy as np
from skimage.measure import find_contours
from skimage.transform import downscale_local_mean
from geometry import distance_p_to_plg, get_segments, distance_p_to_segments_tensor

EPSILON = 1e-5


def log_iou(rendered, target, epsilon=1) -> torch.Tensor:
    true = torch.as_tensor(target)
    pred = torch.as_tensor(rendered)

    I = torch.minimum(true, pred)
    U = torch.maximum(true, pred)
    loss = torch.log(I.sum() + epsilon) - torch.log(U.sum() + epsilon)
    return loss


def boundary(polygon: torch.Tensor, target: np.ndarray) -> torch.Tensor:
    """
    :param polygon: a list of vertices
    :param target: connected components of pixels
    :return: _loss
    """
    B = find_contours(target)[0][:-1]
    # transform coord-array to coord-pixels
    B = B[..., [1, 0]]
    distances = []
    for b in B:
        distances.append(distance_p_to_plg(b, polygon, square=False))
    loss = torch.stack(distances).sum()
    return loss


def orthogonal(polygon: torch.Tensor) -> torch.Tensor:
    start_p, end_p = get_segments(polygon)
    edges_v = end_p - start_p                               # p(i-1) pi
    next_edges_v = torch.roll(edges_v, shifts=1, dims=0)    # pi p(i+1)
    loss = torch.abs(edges_v * next_edges_v).sum()          # \sum_p |p(i-1) pi * pi p(i+1)|
    return loss


def center(edges: Tuple[torch.Tensor, torch.Tensor],
           boundaries: np.ndarray,
           downscale_factor: int = 4,
           ) -> torch.Tensor:
    """
    :param edges:
    :param boundaries: binary image
    :param downscale_factor:
    :return:
    """
    scaled_boundaries = downscale_local_mean(boundaries, factors=(downscale_factor, downscale_factor))
    B = np.argwhere(scaled_boundaries)
    B = B[..., [1, 0]] * downscale_factor
    ds = []
    for b in B:
        distance = distance_p_to_segments_tensor(b, edges)
        ds.append(distance + EPSILON)
    ds = torch.stack(ds)
    loss = ds.sum()
    return loss


def nearby(vertices: torch.Tensor,
           target: np.ndarray | torch.Tensor,
           L: np.ndarray | torch.Tensor
           ) -> torch.Tensor:
    L = torch.as_tensor(L)
    target = torch.as_tensor(target)
    delta = L @ vertices - target
    loss = (delta * delta).sum()
    return loss


def alignment(v_edges: torch.Tensor) -> torch.Tensor:
    loss = torch.amin(torch.abs(v_edges), dim=-1)
    loss = torch.sum(loss)
    return loss
