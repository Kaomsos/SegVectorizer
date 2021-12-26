from __future__ import annotations
from typing import Tuple, List, Dict, TypeVar, Generic, Union
from collections.abc import Collection
from numpy.typing import ArrayLike
from torchtyping import TensorType
from torch import Tensor
import numpy as np


Edge = Tuple[int, int]
EdgeCollection = Collection[Edge]

Vertex = int
VertexCollection = Collection[Vertex]

AdjacentList = Dict[Vertex, List[Vertex]]

PointCollection = np.ndarray
SegmentCollection = Tuple[PointCollection, PointCollection]


Color = Union[int, Tuple[int, int, int]]
Palette = Dict[str, Color]

# sort of "lazy import"
# to deal with circular import
import entity_class
SingleConnectedComponent = entity_class.SingleConnectedComponent
Polygon = entity_class.Polygon
Rectangle = entity_class.Rectangle
Contour = Polygon
WallCenterLine = entity_class.WallCenterLine


Coordinate2D = Tuple[float, float]
Radius = float
Length = float
Vector2D = Tuple[float, float]
