from __future__ import annotations

from PIL import Image
from pathlib import Path

from .entity.image import BinaryImage
import numpy as np


class BinaryImageFromFile(BinaryImage):
    def __init__(self,
                 path: str | Path,
                 scale: float = 1
                 ):
        self._path = path
        self._image = Image.open(path)
        self._image = self._image.resize((int(self._image.size[0] * scale),
                                          int(self._image.size[1] * scale)))
        bin_arr = self._get_array(self._image)
        super(BinaryImageFromFile, self).__init__(bin_arr)

    @staticmethod
    def _get_array(rgb_image, threshold=127):
        array = np.array(rgb_image)[..., :3]
        array = np.where((array <= threshold).all(axis=-1), 1, 0)
        return array.astype("float64")


if __name__ == "__main__":
    path_ = "data/rect2_compress.jpg"
    img = BinaryImageFromFile(path_)
    arr = img.array
