# %%
import torch
from torch.optim import Adam, Adagrad, RMSprop, Adadelta
from torchviz import make_dot

from rasterizer import Base2DPolygonRasterizer, BoundingBox2DRasterizer, FixedCenterRectangle2DRasterizer
from objective import log_iou, boundary, orthogonal
from image_reader import BinaryImageFromFile
import matplotlib.pyplot as plt
import numpy as np

# %%
if __name__ == "__main__":
    pass

