from __future__ import annotations
from typing import Tuple, List, Dict, Set, TYPE_CHECKING
if TYPE_CHECKING:
    from typing_ import WallCenterLine, Rectangle, Palette, Color, SingleConnectedComponent, Contour

import numpy as np

from entity.graph import WallCenterLine
from geometry import find_connected_components
from room_contour_optimization import alternating_optimize as fit_room_contour
from wall_centerline_optimization import alternating_optimize as fit_wall_center_line
from open_points_extraction import fit_open_points, insert_open_points_in_wcl


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

        # max iteration number of optimizations
        self._delta_a = None
        self._delta_b = 10
        self._max_iter_room_alt_opt = 5
        self._max_iter_room_coord_opt = 0

        self._delta_x = None
        self._delta_y = None
        self._downscale = 5
        self._max_iter_wcl_alt_opt = 5
        self._max_iter_wcl_coord_opt = 5

    def _parse_palette(self, config: PaletteConfiguration):
        self._open_colors = set([self._palette.get(cls)
                                 for cls in config.opens if cls in self._palette.keys()])
        self._boundary_colors = set([self._palette.get(cls)
                                     for cls in config.boundaries if cls in self._palette.keys()])
        self._room_colors = set([self._palette.get(cls)
                                 for cls in config.rooms if cls in self._palette.keys()])

    def _vectorize(self, segmentation):
        # init and get hyper_paramenters
        open_cc, boundary_cc, room_cc = self._extract_connected_components(segmentation)
        rects: List[Rectangle] = [self._get_rectangle(o) for o in open_cc]

        self._set_hyper_parameters_by_rectangles(rects)

        # room contour optimization
        room_contours: List[Contour] = [self._get_room_contour(cc) for cc in room_cc]

        # wall center line optimization
        wcl = self._get_wall_center_line(room_contours, boundary_cc)

        wcl_o = insert_open_points_in_wcl(rects, wcl)

        self.rects = rects
        self.boundary = boundary_cc
        self.wcl = wcl

        return wcl_o

    def _extract_connected_components(self, segmentation):
        open_cc: List[SingleConnectedComponent] = []
        for c in self._open_colors:
            open_cc += find_connected_components(segmentation, c, threshold=self._open_threshold)

        # make a copy of open_cc by slicing
        boundary_cc_list: List[SingleConnectedComponent] = open_cc[:]
        for c in self._boundary_colors - self._open_colors:
            boundary_cc_list += find_connected_components(segmentation, c, threshold=self._boundary_threshold)

        boundary_cc = np.zeros_like(boundary_cc_list[0].array, dtype=int)
        for cc in boundary_cc_list:
            boundary_cc |= cc.array

        room_cc: List[SingleConnectedComponent] = []
        for c in self._room_colors:
            room_cc += find_connected_components(segmentation, c, threshold=self._room_threshold)

        return open_cc, boundary_cc, room_cc

    @staticmethod
    def _get_rectangle(open_: SingleConnectedComponent) -> Rectangle:
        rect = fit_open_points(open_)
        return rect

    def _set_hyper_parameters_by_rectangles(self, rects: List[Rectangle]) -> None:
        wall_width = np.array([(rect.w, rect.h) for rect in rects]).min(axis=-1).mean()
        self._delta_a = wall_width * 2
        self._delta_x = wall_width * 2
        self._delta_y = wall_width * 2

    def _get_room_contour(self, cc: SingleConnectedComponent) -> Contour:
        contour = fit_room_contour(cc,
                                   delta_a=self._delta_a,
                                   delta_b=self._delta_b,
                                   max_alt_iter=self._max_iter_room_alt_opt,
                                   max_coord_iter=self._max_iter_room_coord_opt,
                                   )
        return contour

    def _get_wall_center_line(self, contours: List[Contour], boundary: np.ndarray) -> WallCenterLine:
        wcl_new = fit_wall_center_line(WallCenterLine(contours),
                                       delta_x=self._delta_x,
                                       delta_y=self._delta_y,
                                       boundary=boundary,
                                       downscale=self._downscale,
                                       max_alt_iter=self._max_iter_wcl_alt_opt,
                                       max_coord_iter=self._max_iter_wcl_coord_opt,
                                       )
        return wcl_new
