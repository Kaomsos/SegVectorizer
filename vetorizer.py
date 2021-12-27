from __future__ import annotations
from typing import Tuple, List, Dict, Set, TYPE_CHECKING
if TYPE_CHECKING:
    from typing_ import WallCenterLine, Rectangle, Palette, Color, SingleConnectedComponent, Contour

from collections import UserDict
import numpy as np

from entity.graph import WallCenterLine
from geometry import find_connected_components
from room_contour_optimization import alternating_optimize as fit_room_contour
from wall_centerline_optimization import alternating_optimize as fit_wall_center_line
from open_points_extraction import fit_open_points, insert_open_points_in_wcl


class PaletteConfiguration(UserDict):
    """
     an adaptor of palette
    """
    default_door = ("door", )
    default_window = ("window", )
    default_open = ("door&window", )
    default_boundary = ("wall",)
    default_room = ("bedroom", "living room", "kitchen", "dining room")

    def __init__(self,
                 palette: Palette = None,
                 add_door=(),
                 add_window=(),
                 add_open=(),
                 add_boundary=(),
                 add_room=(),
                 ):
        super(PaletteConfiguration, self).__init__()
        if palette is not None:
            self.data.update(palette)

        door: Set[str] = set(self.default_door) | set(add_door)
        window: Set[str] = set(self.default_window) | set(add_window)

        open_: Set[str] = set(self.default_open) | door | window | set(add_open)
        boundary: Set[str] = set(self.default_boundary) | set(open_) | set(add_boundary)
        room: Set[str] = set(self.default_room) | set(add_room)

        self._agg: Dict[str, Set[str]] = {
            "door": door,
            "window": window,
            "open": open_,
            "boundary": boundary,
            "room": room,
        }

    def add_door(self, item: str):
        self._agg["door"].add(item)

        self._agg["open"].add(item)
        self._agg["boundary"].add(item)

    def add_window(self, item: str):
        self._agg["window"].add(item)

        self._agg["open"].add(item)
        self._agg["boundary"].add(item)

    def add_open(self, item: str):
        self._agg["open"].add(item)

        self._agg["boundary"].add(item)

    def add_boundary(self, item: str):
        self._agg["boundary"].add(item)

    def add_room(self, item: str):
        self._agg["room"].add(item)

    def get_colors(self, type_: str) -> Set[Color]:
        keys = self._agg.get(type_, [])
        colors = filter(lambda x: x is not None, [self.data.get(k, None) for k in keys])
        return set(colors)

    @property
    def doors(self) -> Set[Color]:
        return self.get_colors("door")

    @property
    def windows(self) -> Set[Color]:
        return self.get_colors("window")

    @property
    def opens(self) -> Set[Color]:
        return self.get_colors("open")

    @property
    def boundaries(self) -> Set[Color]:
        return self.get_colors("boundary")

    @property
    def rooms(self) -> Set[Color]:
        return self.get_colors("room")


p_config = PaletteConfiguration()


class Vectorizer:
    def __init__(self,
                 palette_config: PaletteConfiguration = p_config,
                 ) -> None:
        self._palette_config = palette_config

        self._door_colors: Set[Color] = None
        self._window_colors: Set[Color] = None
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
        self._max_iter_wcl_alt_opt = 10
        self._max_iter_wcl_coord_opt = 5

    def _parse_palette(self, config: PaletteConfiguration):
        self._door_colors = config.doors
        self._window_colors = config.windows
        self._open_colors = config.opens
        self._boundary_colors = config.boundaries
        self._room_colors = config.rooms

    def _vectorize(self, segmentation):
        # init and get hyper_parameters
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
        self._delta_a = wall_width * 1.5
        self._delta_x = wall_width * 1.5
        self._delta_y = wall_width * 1.5

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
