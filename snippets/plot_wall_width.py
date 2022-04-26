from PIL import Image
import pickle
import numpy as np

from segvec import convert_a_segmentation, Vectorizer, PaletteConfiguration
from segvec.utils import (plot_wcl_against_target,
                          plot_position_of_rects,
                          plot_room_contours,
                          plot_wcl_o_against_target,
                          plot_binary_image,
                          plot_empty_image_like,
                          plot_rooms_in_wcl)
from segvec.utils import palette
from segvec.entity.image import SingleConnectedComponent
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from copy import deepcopy

p = {
    '厨房': -1,
    '阳台': 1,
    '卫生间': 2,
    '卧室': 3,
    '客厅': 4,
    '墙洞': 5,
    '玻璃窗': 6,
    '墙体': 7,
    '书房': 8,
    '储藏间': 9,
    '门厅': 10,
    '其他房间': 11,
    '未命名': 12,
    '客餐厅': 13,
    '主卧': 14,
    '次卧': 15,
    '露台': 16,
    '走廊': 17,
    '设备平台': 18,
    '储物间': 19,
    '起居室': 20,
    '空调': 21,
    '管道': 22,
    '空调外机': 23,
    '设备间': 24,
    '衣帽间': 25,
    '中空': 26
}


def colorize(im, seg, i, color, color_alt=None):
    color = (color[0] / 255, color[1] / 255, color[2] / 255, color[3] / 100)
    arr = deepcopy(im)
    mask = (seg == i)
    for ch, cmp in zip(arr.T, color):
        ch[mask.T] = cmp
    if color_alt is not None:
        color_alt = (color_alt[0] / 255, color_alt[0] / 255, color_alt[0] / 255, color_alt[3] / 100)
        for ch, cmp in zip(arr.T, color_alt):
            ch[mask.T] = cmp
    return arr


if __name__ == "__main__":
    with open("../data/refined_seg.pickle", 'rb') as f:
        seg = pickle.load(f)
    im = np.full(seg.shape+(4, ), 0, dtype=float)

    im = colorize(im, seg, 7, (0, 0, 0, 100))
    im = colorize(im, seg, 6, (0, 0, 0, 50))
    im = colorize(im, seg, 5, (0, 0, 0, 30))

    matplotlib.rcParams['font.sans-serif'] = ["SimHei"]
    plt.imshow(im, interpolation='none')
    plt.axis("off")
    # plt.legend(handles=[Patch(color=[0, 0, 0]), Patch(color=[0.5, 0.5, 0.5]), Patch(color=[1, 0, 0])],
    #            labels=["墙体", "门洞", "窗户"],
    #            loc='best',
    #            bbox_to_anchor=(1.2, 1))
    # plt.savefig("../pics/ww.png", dpi=200)
    plt.show()



