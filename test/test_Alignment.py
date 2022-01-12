from unittest import TestCase
import pickle
from segvec import PaletteConfiguration
from segvec.utils import *
from segvec.entity.wall_center_line import WallCenterLineWithOpenPoints
from segvec.main_steps.alignment import optimize


class TestAlignment(TestCase):
    def _init(self, wclo_path, seg_path):
        with open(seg_path, 'rb') as f:
            self.seg = pickle.load(f)

        with open(wclo_path, 'rb') as f:
            self.wcl_o = pickle.load(f)

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

        self.p_config = PaletteConfiguration(p,
                                        add_door=('墙洞',),
                                        add_window=('玻璃窗',),
                                        add_boundary=('墙体',),
                                        add_room=('厨房', '阳台', '卫生间', '卧室', '客厅', '书房',
                                                  '储藏间', '门厅', '其他房间',
                                                  '未命名', '客餐厅', '主卧', '次卧', '露台', '走廊',
                                                  '设备平台', '储物间', '起居室', '空调', '管道',
                                                  '空调外机', '设备间', '衣帽间', '中空'),
                                        )

    def test_init(self):
        wclo_path = '../data/wcl_o.pickle'
        seg_path = '../data/seg_reduced.pickle'
        self._init(wclo_path, seg_path)

        # put everything together
        plot_empty_image_like(self.seg)
        plot_wcl_o_against_target(self.wcl_o, title='', annotation=False, show=False)
        plot_rooms_in_wcl(self.wcl_o, self.p_config, title="before aligning", contour=False, show=True)

        optimize(self.wcl_o, slanting_tol=20)
        # after optimize
        plot_empty_image_like(self.seg)
        plot_wcl_o_against_target(self.wcl_o, title='', annotation=False, show=False)
        plot_rooms_in_wcl(self.wcl_o, self.p_config, title="after aligning", contour=False, show=True)



