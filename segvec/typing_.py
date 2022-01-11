from __future__ import annotations
from typing import Tuple, List, Dict, TypeVar, Generic, Union
from collections.abc import Collection
import numpy as np

import segvec.entity.wall_center_line
from . import entity


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
WallCenterLine = entity.wall_center_line.WallCenterLine
WallCenterLineWithOpenPoints = entity.wall_center_line.WallCenterLineWithOpenPoints


Coordinate2D = Point = Union[Tuple[float, float], np.ndarray]
Segment = Union[Tuple[Point, Point], np.ndarray]
Radius = float
Length = float
Vector2D = Tuple[float, float]
