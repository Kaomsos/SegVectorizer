# %%
import argparse
from segvec.utils import palette
from segvec.vetorizer import PaletteConfiguration
from segvec.func import convert_an_image, convert_a_segmentation

###########################################
# create a PaletteConfiguration
p_config = PaletteConfiguration(palette)
p_config.add_open("door&window")
for item in ['bathroom/washroom',
             'livingroom/kitchen/dining add_room',
             'bedroom',
             'hall',
             'balcony',
             'closet']:
    p_config.add_room(item)
p_config.add_boundary("wall")


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", "-i", type=str, help="input of the script")
    param = parser.parse_known_args()[0]
    # wclo_path = param.img
    # convert_an_image(wclo_path, palette_config=p_config)
    import pickle
    with open("../data/seg.pickle", 'rb') as f:
        seg = pickle.load(f)

    p = {
        '厨房': 0,
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

    p_config = PaletteConfiguration(p,
                                    add_door=('墙洞',),
                                    add_window=('玻璃窗',),
                                    add_boundary=('墙体',),
                                    add_room=('厨房', '阳台', '卫生间', '卧室', '客厅', '书房',
                                              '储藏间', '门厅', '其他房间',
                                              '未命名', '客餐厅', '主卧', '次卧', '露台', '走廊',
                                              '设备平台', '储物间', '起居室', '空调', '管道',
                                              '空调外机', '设备间', '衣帽间', '中空'),
                                    )
    convert_a_segmentation(seg, p_config)
