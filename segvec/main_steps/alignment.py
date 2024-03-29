from __future__ import annotations
from typing import TYPE_CHECKING, List, Set, Dict

if TYPE_CHECKING:
    from ..typing_ import WallCenterLineWithOpenPoints, WallCenterLine, Edge, Vertex

import numpy as np


def get_adjacent_edges(wcl: WallCenterLine, i: int = None, e: Edge = None) -> Set[Edge]:
    """
    :param e:
    :param i:
    :param wcl:
    :return:
    """
    assert i is not None or e is not None, 'edge are not specified'
    assert not (i is not None and e is not None), 'repeating definition of edge'

    if i is not None:
        e = wcl.i2e[i]
    v1, v2 = e
    adj2v1 = set((v, v1) if v < v1 else (v1, v) for v in wcl.get_adjacent_vertices(v1))
    adj2v2 = set((v, v2) if v < v2 else (v2, v) for v in wcl.get_adjacent_vertices(v2))
    return (adj2v1 | adj2v2) - {e}


def find_connected_edges(adj_edges: Dict[Edge, List[Edge]]) -> List[List[Edge]]:
    visited = {e: False for e in adj_edges.keys()}
    ccs = []

    for e in visited.keys():
        q = []
        cc = []
        q.append(e)
        while q:
            a = q.pop(0)
            if not visited[a]:
                cc.append(a)
                q.extend(adj_edges[a])
                visited[a] = True
        if cc:
            ccs.append(cc)

    return ccs


def get_vertices_from_edges(edges: List[Edge]) -> Set[Vertex]:
    s = set()
    for e in edges:
        s |= set(e)
    return s


def align_vertex(wcl_o: WallCenterLine,
                 slanting_tol: float = 10,
                 ):
    assert 0 <= slanting_tol < 45
    sin_threshold = np.sin(np.pi * slanting_tol / 180)

    # get all edges
    sps, eps = wcl_o.segments_collection
    vecs = sps - eps

    # project on (1, 0) and (0, 1)
    vecs_norm = np.sqrt((vecs * vecs).sum(axis=-1))
    vecs_norm = np.where(vecs_norm > 0, vecs_norm, 1)
    sin_x = np.abs(vecs[..., 1]) / vecs_norm
    sin_y = np.abs(vecs[..., 0]) / vecs_norm

    # category edges into H, V and N three classes
    # NOTE: zero-length edges will be the intersection of H and V
    H_i = np.argwhere(sin_x < sin_threshold).reshape(-1)
    V_i = np.argwhere(sin_y < sin_threshold).reshape(-1)
    i2e = wcl_o.i2e
    H = set(i2e[i] for i in H_i)
    V = set(i2e[i] for i in V_i)

    # find collinear vertices separately in H and V
    H_adj = {e: list(filter(lambda x: x in H, get_adjacent_edges(wcl_o, e=e))) for e in H}
    V_adj = {e: list(filter(lambda x: x in V, get_adjacent_edges(wcl_o, e=e))) for e in V}

    H_cc = find_connected_edges(H_adj)
    V_cc = find_connected_edges(V_adj)

    # set coordinate of these coordinates
    H_cc = [get_vertices_from_edges(cc) for cc in H_cc]
    V_cc = [get_vertices_from_edges(cc) for cc in V_cc]

    for cc in H_cc:
        coords = [wcl_o.get_coordinate_by_v(v) for v in cc]
        mean_ = sum(coords) / len(coords)
        for v, c in zip(cc, coords):
            new_coord = (c[0], mean_[1])
            wcl_o.set_coordinate_by_v(v, new_coord)

    for cc in V_cc:
        coords = [wcl_o.get_coordinate_by_v(v) for v in cc]
        mean_ = sum(coords) / len(coords)
        for v, c in zip(cc, coords):
            new_coord = (mean_[0], c[1])
            wcl_o.set_coordinate_by_v(v, new_coord)

    pass