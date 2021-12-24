from __future__ import annotations
from typing import Tuple, List, Dict
from typing_ import Palette, Color
from geometry import find_connected_components


class Vectorizer:
    def __init__(self,
                 palette: Palette
                 ) -> None:
        self._palette = palette
        self._open: List[Color] = None
        self._boundary: List[Color] = None
        self._room: List[Color] = None
        self._parse_palette()

    def _parse_palette(self):
        self._open = None

    def _vectorize(self, segmentation):
        pass

    def extract_connected_components(self):
        pass

