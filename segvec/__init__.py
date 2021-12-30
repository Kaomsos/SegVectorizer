from . import entity
from . import main_steps
from . import softras
from . import geometry
from . import image_reader
from . import func
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import typing_
from . import utils
from . import vetorizer

from .func import convert_an_image, convert_a_segmentation
from .vetorizer import Vectorizer, PaletteConfiguration
