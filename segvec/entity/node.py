from __future__ import annotations
from typing import Any, TYPE_CHECKING
if TYPE_CHECKING:
    from ..typing_ import Coordinate2D


class Node:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
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

    def __hash__(self):
        return id(self)


class NodeWith2DCoordinate(Node):
    def __init__(self, coordinate: Coordinate2D, **kwargs):
        self._coord = coordinate
        super(NodeWith2DCoordinate, self).__init__(**kwargs)

