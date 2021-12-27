from __future__ import annotations
from typing import Tuple, List, Dict, TypeVar, Generic, Union
from collections.abc import Collection
from numpy.typing import ArrayLike
from torchtyping import TensorType
from torch import Tensor
import numpy as np

import entity


Edge = Tuple[int, int]
EdgeCollection = Collection[Edge]

Vertex = int
VertexCollection = Collection[Vertex]

AdjacentList = Dict[Vertex, List[Vertex]]

PointCollection = np.ndarray
SegmentCollection = Tuple[PointCollection, PointCollection]


Color = Union[int, Tuple[int, int, int]]
Palette = Dict[str, Color]


SingleConnectedComponent = entity.image.SingleConnectedComponent
Polygon = entity.polygon.Polygon
Rectangle = entity.polygon.Rectangle
Contour = Polygon
WallCenterLine = entity.graph.WallCenterLine
WallCenterLineWithOpenPoints = entity.graph.WallCenterLineWithOpenPoints


Coordinate2D = Point = Union[Tuple[float, float], np.ndarray]
Segment = Tuple[Point, Point]
Radius = float
Length = float
Vector2D = Tuple[float, float]
