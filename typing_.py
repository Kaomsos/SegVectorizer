from __future__ import annotations
from typing import Tuple, List, Dict, TypeVar, Generic, Union
from collections.abc import Collection
from numpy.typing import ArrayLike
from torchtyping import TensorType
from torch import Tensor
import numpy as np

# from entity_class import *

Edge = Tuple[int, int]
EdgeCollection = Collection[Edge]

Vertex = int
VertexCollection = Collection[Vertex]

AdjacentList = Dict[Vertex, List[Vertex]]

PointCollection = np.ndarray
SegmentCollection = Tuple[PointCollection, PointCollection]


Color = Union[int, Tuple[int, int, int]]
Palette = Dict[str, Color]
