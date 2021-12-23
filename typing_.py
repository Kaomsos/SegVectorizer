from typing import Tuple, List, Dict, TypeVar, Generic
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
