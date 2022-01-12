from __future__ import annotations
from typing import List, TYPE_CHECKING, Tuple, Any

if TYPE_CHECKING:
    from ..typing_ import EdgeCollection, AdjacentList, Vertex, Edge, Coordinate2D


class Node:
    def __init__(self, **property):
        for k, v in property.items():
            setattr(self, k, v)

    def add_property(self, name: str, value: Any) -> None:
        setattr(self, name, value)

    def remove_property(self, name: str) -> None:
        self.__dict__.pop(name, None)

    def has_property(self, name: str) -> bool:
        return hasattr(self, name)

    @property
    def id(self):
        return id(self)


class NodeWith2DCoordinate(Node):
    def __init__(self, coordinate: Coordinate2D, **property):
        self._coord = coordinate
        super(NodeWith2DCoordinate, self).__init__(**property)


class EdgeMy(Tuple[Node, Node]):
    def __init__(self, pair: Tuple[Node, Node], **property):
        self._pair = pair
        self._v1, self._v2 = self._pair
        for k, v in property.items():
            setattr(self, k, v)


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

    def get_adjacent_vertices(self, i: Vertex):
        return self._adjacency_list[i]

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


