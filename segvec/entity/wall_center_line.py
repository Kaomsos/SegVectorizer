from __future__ import annotations
from typing import TYPE_CHECKING, List, Tuple, Dict

import numpy as np
from scipy.sparse import csr_matrix

from .graph import UndirectedGraph
from .mapping import SemiIdentityMapping
from .polygon import Polygon

if TYPE_CHECKING:
    from typing_ import Vertex, Edge, Coordinate2D, Segment


class WallCenterLine(UndirectedGraph):
    def __init__(self, room_contours: List[Polygon]) -> None:
        coordinates = []
        edges = []
        rooms = []
        start = 0

        ###############################################
        # iterate over all add_room contours (Polygon):
        # 1. gathering all the 2d coordinates
        # 2. construct a UndirectedGraph
        for plg in room_contours:
            coordinates.append(plg.numpy_array)
            interval = len(plg)
            for i in range(interval):
                shift1 = i % interval
                shift2 = (i + 1) % interval
                edges.append((start + shift1, start + shift2))
            rooms.append(list(range(start, start + interval)))
            start += interval

        # Initialize UndirectedGraph
        num_vertices = start
        super(WallCenterLine, self).__init__(num_vertices, edges)

        self._init_n = num_vertices

        # rooms and room_types
        self._rooms: List[List[Vertex]] = rooms
        self._room_types = None

        # Gathering 2d coordinates as a big array
        coordinates = np.concatenate(coordinates, axis=0)
        self._init_coordinates = coordinates
        self._cur_coordinates = coordinates

        ##############################################################
        # defining mappings
        # Sets:
        #       i: a indices of v whose values are a natural sequence
        #       v: all vertices in current adjacency list (graph)
        #       p: all vertices in the initial adjacency list (graph)
        #       j: a indices of p whose values are a natural sequence
        ##############################################################
        # p2j and j2p are determined by the init codes above
        # and are somehow constant
        self._p2j = dict(zip(self._adjacency_list.keys(), range(self._n)))
        self._j2p = dict(zip(range(self._n), self._adjacency_list.keys()))
        # p2v is naturally initialized as SemiIdentityMapping
        self._p2v = SemiIdentityMapping()
        # v2i and i2v are are determined by the init codes above
        # and are variable
        self._v2i = dict(zip(self._adjacency_list.keys(), range(self._n)))
        self._i2v = dict(zip(range(self._n), self._adjacency_list.keys()))

    def _update_v2i_i2v(self):
        self._v2i = dict(zip(self._adjacency_list.keys(), range(self._n)))
        self._i2v = dict(zip(range(self._n), self._adjacency_list.keys()))

    @property
    def sparse_matrix(self) -> csr_matrix:
        # return an adjacency matrix A(V, V)
        rows = []
        cols = []

        # defining a map
        # from the current vertices
        # to their indices by natural sequence
        v2i = self._v2i
        for v1, vl in self._adjacency_list.items():
            for v2 in vl:
                rows.append(v2i[v1])
                cols.append(v2i[v2])
        size = len(rows)
        m = csr_matrix(([1] * size, (rows, cols)), shape=(self._n, self._n))
        return m

    @property
    def matrix(self) -> np.ndarray:
        m = self.sparse_matrix
        return m.toarray()

    @property
    def segments_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        # return a matrix S(e, v) and E(e, v)
        # such that given current coordinates array P(v, 2)
        # S(E, V) @ P(v, 2) will return an array of all current start points coordinates, and
        # E(E, V) @ P(v, 2) will return an array end points coordinates

        edges = self.edges

        n_edge = len(edges)
        n_vertex = self._n

        s_i, e_i = zip(*edges)
        s_i = [self._v2i[v] for v in s_i]
        e_i = [self._v2i[v] for v in e_i]
        range_i = np.arange(n_edge)

        S = np.zeros((n_edge, n_vertex), dtype=int)
        E = np.zeros((n_edge, n_vertex), dtype=int)
        S[range_i, s_i] = 1
        E[range_i, e_i] = 1
        return S, E

    @property
    def L(self) -> np.ndarray:
        # a map from initial P to current V
        j2i = {j: self._v2i[self._p2v[p]] for j, p in self._j2p.items()}
        size = len(j2i)
        m = csr_matrix(([1] * size, (list(j2i.keys()), list(j2i.values()))), shape=(self._init_n, self._n))
        return m.toarray()

    @property
    def V(self) -> np.ndarray:
        return self._cur_coordinates

    @property
    def P(self) -> np.ndarray:
        return self._init_coordinates

    @property
    def i2v(self) -> Dict[int, Vertex]:
        return self._i2v

    @property
    def i2e(self) -> Dict[int, Edge]:
        edges = self.edges
        n_edges = len(edges)
        mapping = dict(zip(range(n_edges), edges))
        return mapping

    @property
    def e2i(self):
        edges = self.edges
        n_edges = len(edges)
        mapping = dict(zip(edges, range(n_edges)))
        return mapping

    @property
    def segments_collection(self) -> Tuple[np.ndarray, np.ndarray]:
        S, E = self.segments_matrix
        S_p, E_p = S @ self._cur_coordinates, E @ self._cur_coordinates
        return S_p, E_p

    @property
    def rooms(self) -> List[np.ndarray]:
        coord = [self._get_coordinates_by_ps(ps) for ps in self._rooms]
        return coord

    @property
    def room_types(self) -> List[int]:
        return self._room_types

    @room_types.setter
    def room_types(self, val: List[int]):
        self._room_types = val

    def get_coordinate_by_v(self, v: Vertex) -> Coordinate2D:
        i = self._v2i[v]
        return self._cur_coordinates[i]

    def set_coordinate_by_v(self, v: Vertex, value: Coordinate2D) -> None:
        i = self._v2i[v]
        self._cur_coordinates[i] = np.array(value)

    def _get_coordinates_by_ps(self, ps: List[Vertex]):
        indices = [self._v2i[self._p2v[p]] for p in ps]
        return self._cur_coordinates[indices]

    def get_coordinates_by_e(self, e: Edge) -> Segment:
        i, j = e
        c1 = self.get_coordinate_by_v(i)
        c2 = self.get_coordinate_by_v(j)
        return c1, c2

    def merge_vertices(self, i: Vertex, j: Vertex):
        update_which = self._v2i[i]
        remove_which = self._v2i[j]
        if i == j:
            return
        indices = list(range(self._n))

        super(WallCenterLine, self).merge_vertices(i, j)

        self._cur_coordinates[update_which] = (self._cur_coordinates[update_which]
                                               + self._cur_coordinates[remove_which]) / 2

        indices.pop(remove_which)
        self._cur_coordinates = self._cur_coordinates[indices]

        self._p2v[j] = i
        self._update_v2i_i2v()

    def append_vertex_to_edge(self, v: Vertex, e: Edge):
        if v in e:
            return
        super(WallCenterLine, self).append_vertex_to_edge(v, e)

        for room in self._rooms:
            i = self._find_insertion_in_room(e, room)
            if i is not None:
                room.insert(i, v)

    def _find_insertion_in_room(self, e: Edge, room: List[Vertex]):
        n = len(room)
        for i in range(n):
            re1 = self._p2v[room[i % n]]
            re2 = self._p2v[room[(i + 1) % n]]
            if re1 == e[0] and re2 == e[1]:
                return i + 1
            if re1 == e[1] and re2 == e[0]:
                return i + 1
        return None

    def set_current_coordinates(self, array: np.ndarray):
        self._cur_coordinates = array


class WallCenterLineWithOpenPoints(WallCenterLine):
    def __init__(self):
        self._open_edge = []
        self._door_edge = []
        self._window_edge = []

    @property
    def opens(self) -> List[np.ndarray]:
        return [np.stack(self.get_coordinates_by_e(e)) for e in self._open_edge]

    @property
    def doors(self) -> List[np.ndarray]:
        return [np.stack(self.get_coordinates_by_e(e)) for e in self._door_edge]

    @property
    def windows(self) -> List[np.ndarray]:
        return [np.stack(self.get_coordinates_by_e(e)) for e in self._window_edge]

    @property
    def walls(self) -> List[np.ndarray]:
        return [np.stack(self.get_coordinates_by_e(e)) for e in set(self.edges) - set(self._open_edge)]

    @property
    def json(self):
        coord = self._cur_coordinates.tolist()

        vertices = list(self._adjacency_list.keys())
        v2c = {v: coord[self._v2i[v]] for v in vertices}
        opens = list(filter(lambda x: x[0] <= x[1], self._open_edge))
        doors = list(filter(lambda x: x[0] <= x[1], self._door_edge))
        windows = list(filter(lambda x: x[0] <= x[1], self._window_edge))
        walls = list(set(self.edges) - set(self._open_edge))

        rooms = [[self._p2v[p] for p in room] for room in self._rooms]
        i2r = dict(enumerate(rooms))

        json = {
            "vertices": {
                "all": vertices,
                "coordinate": v2c,
            },

            "edges": {
                "door": doors,
                "window": windows,
                "wall": walls,
            },

            "rooms": {
                "all":  list(i2r.keys()),
                "contour": i2r,
            },
        }
        return json

    @classmethod
    def from_wcl(cls, wcl: WallCenterLine):
        inst = cls()
        inst.__dict__.update(wcl.__dict__)
        return inst

    def add_vertex(self, coord: Coordinate2D) -> Vertex:
        v = super(WallCenterLineWithOpenPoints, self).add_vertex()
        self._cur_coordinates = np.vstack((self._cur_coordinates, coord))
        self._update_v2i_i2v()
        return v

    # TODO: to merge overlapping vertices
    def insert_open_to_edge(self, seg: Segment, e: Edge) -> Edge:
        self._check_edge_exists(e)
        e1 = self.get_coordinate_by_v(e[0])
        e2 = self.get_coordinate_by_v(e[1])

        p1, p2 = seg
        v1 = self.add_vertex(p1)
        v2 = self.add_vertex(p2)
        self.connect_vertices(v1, v2)

        # use L1 distance for comparing
        if np.abs(p1 - e1).sum() < np.abs(p2 - e1).sum():
            to_e1 = v1
        else:
            to_e1 = v2

        if np.abs(p1 - e2).sum() < np.abs(p2 - e2).sum():
            to_e2 = v1
        else:
            to_e2 = v2

        self.connect_vertices(e[0], to_e1)
        self.connect_vertices(e[1], to_e2)
        self.disconnect_vertices(*e)

        self._add_to_opens((v1, v2))

        return v1, v2

    def insert_window_to_edge(self, seg: Segment, e: Edge) -> Edge:
        v1, v2 = self.insert_open_to_edge(seg, e)
        self._add_to_windows((v1, v2))
        return v1, v2

    def insert_door_to_edge(self, seg: Segment, e: Edge) -> Edge:
        v1, v2 = self.insert_open_to_edge(seg, e)
        self._add_to_doors((v1, v2))
        return v1, v2

    def _add_to_opens(self, e: Edge):
        self._check_edge_exists(e)
        self._add_edge_to_list(e, self._open_edge)

    def _add_to_windows(self, e: Edge):
        self._check_edge_exists(e)
        self._add_edge_to_list(e, self._window_edge)

    def _add_to_doors(self, e: Edge):
        self._check_edge_exists(e)
        self._add_edge_to_list(e, self._door_edge)

    @staticmethod
    def _add_edge_to_list(e: Edge, l: list):
        l.append(e)
        l.append(e[::-1])

