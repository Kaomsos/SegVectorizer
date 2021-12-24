from __future__ import annotations
from typing import Tuple, List, Dict, Set
from typing_ import Palette, Color, SingleConnectedComponent
from geometry import find_connected_components


class PaletteConfiguration:
    default_open = ("door", "window", "door&window")
    default_boundary = ("wall",)
    default_room = ("bedroom", "living room", "kitchen", "dining room")

    def __init__(self,
                 add_open=(),
                 add_boundary=(),
                 add_room=(),
                 ):
        self._open: Set[str] = set(self.default_open) | set(add_open)
        self._boundary: Set[str] = set(self.default_boundary) | set(self._open) | set(add_boundary)
        self._room: Set[str] = set(self.default_room) | set(add_room)

    def add_open(self, item: str):
        self._open.add(item)

    def add_boundary(self, item: str):
        self._boundary.add(item)

    def add_room(self, item: str):
        self._room.add(item)

    @property
    def opens(self):
        return set(self._open)

    @property
    def boundaries(self):
        return set(self._boundary)

    @property
    def rooms(self):
        return set(self._room)


p_config = PaletteConfiguration()


class Vectorizer:
    def __init__(self,
                 palette: Palette,
                 palette_config: PaletteConfiguration = p_config,
                 ) -> None:
        self._palette = palette
        self._palette_config = palette_config

        self._open_colors: Set[Color] = None
        self._boundary_colors: Set[Color] = None
        self._room_colors: Set[Color] = None

        self._parse_palette(self._palette_config)

        # threshold of valid connected components
        self._open_threshold = 4
        self._boundary_threshold = 4
        self._room_threshold = 10

    def _parse_palette(self, config: PaletteConfiguration):
        self._open_colors = set([self._palette.get(cls)
                                 for cls in config.opens if cls in self._palette.keys()])
        self._boundary_colors = set([self._palette.get(cls)
                                     for cls in config.boundaries if cls in self._palette.keys()])
        self._room_colors = set([self._palette.get(cls)
                                 for cls in config.rooms if cls in self._palette.keys()])

    def _vectorize(self, segmentation):
        open_cc, boundary_cc, room_cc = self._extract_connected_components(segmentation)
        pass

    def _extract_connected_components(self, segmentation):
        open_cc: List[SingleConnectedComponent] = []
        for c in self._open_colors:
            open_cc += find_connected_components(segmentation, c, threshold=self._open_threshold)

        # make a copy of open_cc by slicing
        boundary_cc: List[SingleConnectedComponent] = open_cc[:]
        for c in self._boundary_colors - self._open_colors:
            boundary_cc += find_connected_components(segmentation, c, threshold=self._boundary_threshold)

        room_cc: List[SingleConnectedComponent] = []
        for c in self._boundary_colors:
            room_cc += find_connected_components(segmentation, c, threshold=self._room_threshold)

        return open_cc, boundary_cc, room_cc
