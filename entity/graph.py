from __future__ import annotations
from typing import List, TYPE_CHECKING, Tuple, Dict

import numpy as np
from scipy.sparse import csr_matrix

from .polygon import Polygon
from .mapping import SemiIdentityMapping

if TYPE_CHECKING:
    from typing_ import EdgeCollection, AdjacentList, Vertex, Edge, Coordinate2D, Segment


# TODO: add self-implemented nodes and edge classes
class UndirectedGraph:
    # adjacency list representation is implemented
    # for the sparsity of wall center lines
    def __init__(self,
                 num_vertices: int,
                 edges: EdgeCollection,
                 ) -> None:

        self._n = num_vertices
        self._next_vertex = num_vertices
        ##############################
        # construct a adjacency list
        self._adjacency_list = self._adjacent_list_from_edges(edges)

        cur_v_num = len(self._adjacency_list)
        assert num_vertices >= cur_v_num, 'Too many vertices are contained in the edges'

        # isolated vertices are encoded by natural sequence
        if num_vertices > cur_v_num:
            max_ = max(self._adjacency_list.keys())
            for i in range(max_, max_ + (num_vertices - cur_v_num)):
                i += 1
                self._adjacency_list[i] = []

    @staticmethod
    def _adjacent_list_from_edges(edges: EdgeCollection) -> AdjacentList:
        adj_list = {}
        for e in edges:
            v1, v2 = e
            adj_list[v1] = adj_list.get(v1, []) + [v2]
            adj_list[v2] = adj_list.get(v2, []) + [v1]
        # removing duplicates
        for k in adj_list.keys():
            adj_list[k] = list(set(adj_list[k]))

        return adj_list

    @property
    def vertices(self) -> List[Vertex]:
        # this property may be useless
        return list(self._adjacency_list.keys())

    @property
    def edges(self) -> List[Edge]:
        # this property may not be used
        # (v1, v2), (v2, v1) will not appear simultaneously
        # and v1 is always lesser than v2
        edges = []
        for v, vl in self._adjacency_list.items():
            for v_ in vl:
                if v < v_:
                    edges.append((v, v_))
        return edges

    def merge_vertices(self, i: Vertex, j: Vertex):
        self._check_vertex_exists(i)
        self._check_vertex_exists(j)
        if i == j:
            return
        vertices_to_j = self._adjacency_list[j]
        for k in vertices_to_j:
            self.connect_vertices(k, i)

        self.remove_vertex(j)

    def append_vertex_to_edge(self, v: Vertex, e: Edge):
        self._check_vertex_exists(v)
        self._check_edge_exists(e)
        if v in e:
            return
        # replace (v1, v2) with (v1, v) and (v2, v)
        v1 = e[0]
        v2 = e[1]

        self.connect_vertices(v, v1)
        self.connect_vertices(v, v2)
        self.disconnect_vertices(v1, v2)

    def insert_vertex_to_edge(self, e: Edge):
        v = self.add_vertex()
        self.append_vertex_to_edge(v, e)

    def add_vertex(self) -> Vertex:
        old = self._next_vertex
        self._adjacency_list[old] = []
        self._next_vertex += 1
        self._n += 1
        return old

    def remove_vertex(self, v: Vertex):
        self._check_vertex_exists(v)

        connected_vertices = self._adjacency_list.pop(v)
        for cv in connected_vertices:
            self._adjacency_list[cv] = [k for k in self._adjacency_list[cv] if k != v]

        self._n -= 1

    def connect_vertices(self, i: Vertex, j: Vertex):
        """
        add an edge
        :param i:
        :param j:
        :return:
        """
        self._check_vertex_exists(i)
        self._check_vertex_exists(j)
        if i == j or self._edge_exists((i, j)):
            return

        self._adjacency_list[i] += [j]
        self._adjacency_list[j] += [i]

    def disconnect_vertices(self, i: Vertex, j: Vertex):
        """
        remove an edge
        :param i:
        :param j:
        :return:
        """
        self._check_vertex_exists(i)
        self._check_vertex_exists(j)
        if i == j or not self._edge_exists((i, j)):
            return

        self._adjacency_list[i] = [k for k in self._adjacency_list[i] if k != j]
        self._adjacency_list[j] = [k for k in self._adjacency_list[j] if k != i]

    def _vertex_exists(self, v: Vertex) -> bool:
        return v in self._adjacency_list.keys()

    def _check_vertex_exists(self, v: Vertex):
        assert self._vertex_exists(v)

    def _edge_exists(self, e: Edge) -> bool:
        return self._vertex_exists(e[0])\
               and self._vertex_exists(e[1])\
               and e[0] in self._adjacency_list[e[1]]\
               and e[1] in self._adjacency_list[e[0]]

    def _check_edge_exists(self, e: Edge):
        assert self._edge_exists(e)


class WallCenterLine(UndirectedGraph):
    def __init__(self, room_contours: List[Polygon]) -> None:
        coordinates = []
        edges = []
        start = 0

        ###############################################
        # iterate over all add_room contours (Polygon):
        # 1. gathering all the 2d coordinates
        # 2. construct a UndirectedGraph
        for c in room_contours:
            coordinates.append(c.numpy_array)
            interval = len(c)
            for i in range(interval):
                shift1 = i % interval
                shift2 = (i + 1) % interval
                edges.append((start + shift1, start + shift2))
            start += interval

        # Initialize UndirectedGraph
        num_vertices = start
        self._init_n = num_vertices
        super(WallCenterLine, self).__init__(num_vertices, edges)

        # Gathering 2d coordinates as a big array
        coordinates = np.concatenate(coordinates, axis=0)
        self._init_coordinates = coordinates
        self._cur_coordinates = coordinates

        ###############################
        # defining mappings
        # Sets:
        #       i: a indices of v whose values are a natural sequence
        #       v: all vertices in current adjacency list (graph)
        #       p: all vertices in the initial adjacency list (graph)
        #       j: a indices of p whose values are a natural sequence
        ###############################
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
        # return a matrix S(E, V) and E(E, V)
        # such that given current coordinates array P(V, 2)
        # (S @ P)(E, 2) will return an array of all current start point coordinates
        # so is the matrix E(E, V)

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
    def segments_collection(self) -> Tuple[np.ndarray, np.ndarray]:
        S, E = self.segments_matrix
        S_p, E_p = S @ self._cur_coordinates, E @ self._cur_coordinates
        return S_p, E_p

    def get_coordinate_by_v(self, v: Vertex) -> np.ndarray | Coordinate2D:
        i = self._v2i[v]
        return self._cur_coordinates[i]

    def get_coordinates_by_e(self, e: Edge) -> np.ndarray | Segment:
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

    def _update_v2i_i2v(self):
        self._v2i = dict(zip(self._adjacency_list.keys(), range(self._n)))
        self._i2v = dict(zip(range(self._n), self._adjacency_list.keys()))

    def append_vertex_to_edge(self, v: Vertex, e: Edge):
        if v in e:
            return
        super(WallCenterLine, self).append_vertex_to_edge(v, e)

    def set_current_coordinates(self, array: np.ndarray):
        self._cur_coordinates = array


class WallCenterLineWithOpenPoints(WallCenterLine):
    def __init__(self):
        self._open_edge = []
        self._door_edge = []
        self._window_edge = []

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

