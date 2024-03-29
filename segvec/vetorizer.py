from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import List, Dict, Set
    from typing_ import (WallCenterLine, Rectangle, Palette, Color,
                         SingleConnectedComponent, Contour, WallCenterLineWithOpenPoints)

from collections import UserDict
import numpy as np
from pathlib import Path
from PIL import Image

from .entity.wall_center_line import WallCenterLine
from .geometry import find_connected_components
from .main_steps.room_contour import alternating_optimize as fit_room_contour
from .main_steps.wall_center_line import alternating_optimize as fit_wall_center_line
from .main_steps.open_points import fit_open_points, insert_open_points_in_wcl
from .main_steps.room_type import refine_room_types
from .main_steps.alignment import align_vertex as enhance_alignment
from .main_steps.wall_width import get_two_means, get_width, WallWidthSolver


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

        # threshold for alignment
        self._slanting_tolerance = 20

        # TODO: convert the following attributes to local variables
        # threshold and representative value for  wall's width
        self._thick_wall = None
        self._thin_wall = None
        self._width_max = None
        self._width_min = None

    @property
    def wall_threshold(self):
        if self._thin_wall is not None and self._thick_wall is not None:
            return (self._thin_wall + self._thick_wall) / 2
        else:
            return None

    @property
    def thin_wall(self):
        # display width of thin wall
        if self.wall_threshold is not None:
            return self.wall_threshold * 2 / 3
        else:
            return None

    @property
    def thick_wall(self):
        # display width of thick wall
        if self.wall_threshold is not None:
            return self.wall_threshold * 4 / 3
        else:
            return None

    def _parse_palette(self, config: PaletteConfiguration):
        self._door_colors = config.doors
        self._window_colors = config.windows
        self._open_colors = config.opens
        self._boundary_colors = config.boundaries
        self._room_colors = config.rooms

    def __call__(self,
                 segmentation: np.ndarray = None,
                 path: str | Path = None
                 ) -> WallCenterLineWithOpenPoints:
        if segmentation is None and path is not None:
            img = Image.open(path)
            segmentation = np.array(img)
        elif segmentation is None and path is None:
            raise ValueError('not enough parameters to define a segmentation')

        wcl_o = self._vectorize(segmentation)
        return wcl_o

    def _vectorize(self, segmentation) -> WallCenterLineWithOpenPoints:
        # init and get hyper_parameters
        open_cc, boundary_cc, room_cc = self.extract_connected_components(segmentation)
        rects: List[Rectangle] = self.get_rectangles(open_cc)

        self.set_hyper_parameters_by_rectangles(rects)

        # room contour optimization
        room_contours: List[Contour] = self.get_room_contours(room_cc)

        # wall center line optimization
        wcl = self.get_wall_center_line(room_contours, boundary_cc)
        self.enhance_alignment(wcl)

        # open points extraction
        wcl_o = self.insert_open_points_in_wcl(opens=rects, wcl=wcl)

        # room type refinement
        wcl_o.room_types = self.get_room_type(wcl_o, segmentation)

        self.rects = rects
        self.boundary = boundary_cc
        self.wcl = wcl

        return wcl_o

    def extract_connected_components(self, segmentation):
        window_cc: List[SingleConnectedComponent] = []
        for c in self._window_colors:
            window_cc += find_connected_components(segmentation, c, threshold=self._boundary_threshold, tag="window")

        door_cc: List[SingleConnectedComponent] = []
        for c in self._door_colors:
            door_cc += find_connected_components(segmentation, c, threshold=self._boundary_threshold, tag="door")

        ########################################################
        # only open_cc will be returned though door and windows are
        open_cc: List[SingleConnectedComponent] = window_cc + door_cc
        for c in self._open_colors - (self._window_colors | self._door_colors):
            open_cc += find_connected_components(segmentation, c, threshold=self._open_threshold)

        ###########################################################
        # boundary is the union of open cc and wall cc
        # make a copy of open_cc by slicing
        boundary_cc_list: List[SingleConnectedComponent] = open_cc[:]
        # boundary_cc_list: List[SingleConnectedComponent] = []
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
    def _get_rectangle(open_: SingleConnectedComponent, fitter=None) -> Rectangle:
        rect = fit_open_points(open_, fitter)
        rect.tag = open_.tag
        return rect

    def get_rectangles(self, opens: List[SingleConnectedComponent], fitter=None) -> List[Rectangle]:
        return [self._get_rectangle(o, fitter) for o in opens]

    def set_hyper_parameters_by_rectangles(self, rects: List[Rectangle]) -> None:
        wall_widths = np.array([(rect.w, rect.h) for rect in rects]).min(axis=-1)
        # w = wall_widths.mean()
        w = np.median(wall_widths)
        self._delta_a = w * 1.5
        self._delta_x = w * 1.5
        self._delta_y = w * 1.5

        self._thin_wall, self._thick_wall = get_two_means(wall_widths)
        self._width_max, self._width_min = max(wall_widths), min(wall_widths)

    def _get_room_contour(self, cc: SingleConnectedComponent) -> Contour:
        contour = fit_room_contour(cc,
                                   delta_a=self._delta_a,
                                   delta_b=self._delta_b,
                                   max_alt_iter=self._max_iter_room_alt_opt,
                                   max_coord_iter=self._max_iter_room_coord_opt,
                                   )
        return contour

    def get_room_contours(self, rooms: List[SingleConnectedComponent]) -> List[Contour]:
        return [self._get_room_contour(cc) for cc in rooms]

    def get_wall_center_line(self, contours: List[Contour], boundary: np.ndarray) -> WallCenterLine:
        wcl_new = fit_wall_center_line(WallCenterLine(contours),
                                       delta_x=self._delta_x,
                                       delta_y=self._delta_y,
                                       boundary=boundary,
                                       downscale=self._downscale,
                                       max_alt_iter=self._max_iter_wcl_alt_opt,
                                       max_coord_iter=self._max_iter_wcl_coord_opt,
                                       )
        return wcl_new

    def enhance_alignment(self, wcl: WallCenterLine):
        enhance_alignment(wcl, self._slanting_tolerance)

    @staticmethod
    def insert_open_points_in_wcl(opens: List[Rectangle], wcl: WallCenterLine) -> WallCenterLineWithOpenPoints:
        return insert_open_points_in_wcl(opens, wcl)

    def get_room_type(self, wcl: WallCenterLine, segmentation: np.ndarray):
        room_types = refine_room_types(wcl,
                                       segmentation,
                                       boundary=self._boundary_colors,
                                       background=(),
                                       )
        return room_types

    def get_wall_width(self, boundary: np.ndarray, wcl: WallCenterLine):
        widths = get_width(target=boundary, wcl=wcl, thin_thick=(self.thin_wall, self.thick_wall), boundary=(self._width_min, self._width_max))
        return widths

    def set_widths_of_wcl(self, wcl: WallCenterLine, boundary: np.ndarray):
        for edge in wcl.edges:
            s = wcl.get_coordinate_by_v(edge[0])
            e = wcl.get_coordinate_by_v(edge[1])

            solver = WallWidthSolver(target=boundary, edge=(s, e), optimizer='exhaustive', boundary=(self._width_min, self._width_max))
            w1, w2 = solver.solve()
            n_v = solver.normal_vector

            delta = n_v * (w1 - w2) / 2

            wcl.set_width_by_e(edge, w1 + w2)
            wcl.set_coordinate_by_v(edge[0], s + delta)
            wcl.set_coordinate_by_v(edge[1], e + delta)
