from __future__ import annotations
from typing import Tuple, List, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from typing_ import EdgeCollection, AdjacentList, Vertex, Edge

import numpy as np
import torch
from skimage.measure import find_contours, approximate_polygon
from scipy.sparse import csr_matrix
import importlib


class BinaryImage:
    def __init__(self,
                 bin_array: np.array,
                 ) -> None:
        self.array = bin_array
        self.pos = 1

    @property
    def size(self) -> Tuple[int, int]:
        return self.array.shape[1], self.array.shape[0]

    @property
    def shape(self) -> Tuple[int, int]:
        return self.array.shape

    @property
    def bbox(self):
        indices: np.ndarray = np.argwhere(self.array == self.pos)
        # array-coordinate to pixel-coordinates
        y_min, x_min = np.min(indices, axis=0).tolist()
        y_max, x_max = np.max(indices, axis=0).tolist()
        return (x_min, y_min), (x_max, y_max)

    @property
    def center(self) -> Tuple[float, float]:
        # array-coordinate
        (x_min, y_min), (x_max, y_max) = self.bbox
        return (x_min + x_max) / 2, (y_min + y_max) / 2

    @property
    def width_height(self) -> Tuple[float, float]:
        (x_min, y_min), (x_max, y_max) = self.bbox
        width = x_max - x_min
        height = y_max - y_min
        return width, height


class SingleConnectedComponent(BinaryImage):
    def __new__(cls,
                bin_array: np.ndarray,
                threshold: float = 10
                ) -> Optional[SingleConnectedComponent]:
        inst = super().__new__(cls)
        super(SingleConnectedComponent, inst).__init__(bin_array)

        # filter trivial pixels
        if threshold > 1:
            (x_min, y_min), (x_max, y_max) = inst.bbox
            if max(x_max - x_min, y_max - y_min) < threshold:
                return None

        # init contours
        inst._init_contours(filter_factor=0.9)
        contours = inst.contours

        # try to return a SingleConnectedComponent instance
        if len(contours) == 1:
            return inst
        elif len(contours) > 1:
            warnings = importlib.import_module('warnings')
            warnings.warn("more than one connected components are found, "
                          "but only the first components are considered")
            return inst
        else:
            # construction failed,
            # return None
            return

    def __init__(self, bin_array, threshold=10):
        pass

    def _init_contours(self, filter_factor=0.9):
        # core function
        contours = find_contours(self.array, fully_connected='high')    # coord = array

        # IMPORTANT:
        # transform from array-coord to pixel-coord
        for i in range(len(contours)):
            contours[i] = contours[i][..., [1, 0]]

        # filter contours that's trivial
        (x_min, y_min), (x_max, y_max) = self.bbox
        threshold = filter_factor * min(x_max - x_min, y_max - y_min)
        contours = self.filter_contours(contours, threshold)

        self.contours = contours

    @staticmethod
    def filter_contours(contours: List[np.array], threshold: float = 10):
        # solving the problem of circular import
        mod = importlib.import_module("geometry")
        get_bounding_box = getattr(mod, 'get_bounding_box')

        # define an inner function
        def is_trivial(contour):
            lt, rb = get_bounding_box(torch.as_tensor(contour))
            delta = torch.abs(lt - rb)

            # a contour is trivial iff
            # its x-interval ans y-interval
            # are both smaller than the threshold
            return (delta < threshold).all()

        filtered = [c for c in contours if not is_trivial(c)]
        return filtered

    @property
    def boundary(self) -> np.ndarray:
        return self.contours[0][: -1]


class Polygon:
    def __init__(self,
                 connected_component: SingleConnectedComponent = None,
                 arr: np.ndarray = None,
                 vertices: np.ndarray = None,
                 tol: float | int = None,
                 ) -> None:
        """

        :param arr: np.ndarray that represents a binary image
        :param vertices: np.ndarray that represents a list vertices
        """
        self._tol = tol if tol is not None else 30
        # extract polygon's vertices
        # from connected pixels component
        vertices_ = None
        if connected_component is not None:
            assert isinstance(connected_component, SingleConnectedComponent)
            vertices_ = approximate_polygon(connected_component.boundary, tolerance=self._tol)

        elif arr is not None:
            contours = find_contours(arr)    # delete the last point
            assert len(contours) <= 1, 'more than one connected component are detected'
            assert len(contours) > 0, 'no connected component is detected'
            vertices_ = approximate_polygon(contours[0], tolerance=self._tol)
            vertices_ = vertices_[..., [1, 0]]
        # directly assign vertices
        elif vertices_ is None:
            vertices_ = vertices

        self._vertices = None
        self.set_vertices(vertices_)
        assert self._vertices is not None, 'no definition of Polygon'

    def __len__(self):
        return self._vertices.shape[0] - 1

    def __getitem__(self, item):
        assert isinstance(item, int)
        assert item < len(self)

        return self._vertices[item]

    def set_vertices(self, vertices: np.ndarray):
        assert len(vertices) > 0, 'empty array'
        assert len(vertices.shape) == 2, 'not a 2d array'
        assert vertices.shape[1] == 2, 'not a array of 2d point'

        # constraints that self._vertices are "closed"
        # IMPROVE: replace indexing implementation
        if (vertices[0] != vertices[-1]).any():
            idx = list(range(vertices.shape[0]))
            self._vertices = vertices[idx + [0]]
        else:
            self._vertices = vertices

    @property
    def plot_x(self):
        return self._vertices[..., 0]

    @property
    def plot_y(self):
        return self._vertices[..., 1]

    @property
    def torch_tensor(self):
        return torch.tensor(self._vertices[:-1], requires_grad=True)

    @property
    def numpy_array(self):
        return self._vertices[:-1]


#################################################
class UndirectedGraph:
    # adjacency list representation is implemented
    # for the sparsity of wall center lines
    def __init__(self,
                 num_vertices: int,
                 edges: EdgeCollection,
                 ) -> None:

        self._n = num_vertices
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

    def merge_vertices(self, i, j):
        assert i in self._adjacency_list.keys()
        assert j in self._adjacency_list.keys()
        if i == j:
            return
        # merge i, j into i
        vertices_to_j = self._adjacency_list.pop(j)
        self._adjacency_list[i] += vertices_to_j
        # removing duplicates
        self._adjacency_list[i] = list(set(self._adjacency_list[i]))

        for k in vertices_to_j:
            # replace all j with i
            self._adjacency_list[k] = [v if v != j else i for v in self._adjacency_list[k]]
            # removing duplicates
            self._adjacency_list[k] = list(set(self._adjacency_list[k]))
        # decrease the number of vertices
        self._n -= 1

    def append_vertex_to_edge(self, v: Vertex, e: Edge):
        assert v in self._adjacency_list.keys()
        self._check_edge_exists(e)
        if v in e:
            return
        # replace (v1, v2) with (v1, v) and (v2, v)
        v1 = e[0]
        v2 = e[1]
        self._adjacency_list[v1] = [x if x != v2 else v for x in self._adjacency_list[v1]]
        self._adjacency_list[v2] = [x if x != v1 else v for x in self._adjacency_list[v2]]

        self._adjacency_list[v].append(v1)
        self._adjacency_list[v].append(v2)
        # removing duplicates
        self._adjacency_list[v1] = list(set(self._adjacency_list[v1]))
        self._adjacency_list[v2] = list(set(self._adjacency_list[v2]))
        self._adjacency_list[v] = list(set(self._adjacency_list[v]))

    def _check_edge_exists(self, e: Edge):
        assert e[0] in self._adjacency_list.keys()
        assert e[1] in self._adjacency_list.keys()
        assert e[0] in self._adjacency_list[e[1]]
        assert e[1] in self._adjacency_list[e[0]]
        return True


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
        # p2v is naturally initialized
        self._p2v = dict(zip(self._adjacency_list.keys(), self._adjacency_list.keys()))
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
    def V(self):
        return self._cur_coordinates

    @property
    def P(self):
        return self._init_coordinates

    @property
    def i2v(self):
        return self._i2v

    @property
    def i2e(self):
        edges = self.edges
        n_edges = len(edges)
        mapping = dict(zip(range(n_edges), edges))
        return mapping

    @property
    def segments_collection(self) -> Tuple[np.ndarray, np.ndarray]:
        S, E = self.segments_matrix
        S_p, E_p = S @ self._cur_coordinates, E @ self._cur_coordinates
        return S_p, E_p

    def merge_vertices(self, i, j):
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
        self._v2i = dict(zip(self._adjacency_list.keys(), range(self._n)))
        self._i2v = dict(zip(range(self._n), self._adjacency_list.keys()))

    def append_vertex_to_edge(self, v: Vertex, e: Edge):
        if v in e:
            return
        super(WallCenterLine, self).append_vertex_to_edge(v, e)

    def set_current_coordinates(self, array: np.ndarray):
        self._cur_coordinates = array




