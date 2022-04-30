import json
import pickle
from segvec.entity.wall_center_line import WallCenterLineWithOpenPoints
from pathlib import Path

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

reverse_p = dict(zip(p.values(), p.keys()))


def load_wcl_o(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


def save_json(obj, path):
    with open(path, 'w', encoding='utf8') as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def serialize_all():
    root = Path('raw_demo')
    for path in root.iterdir():
        if path.is_file() and path.suffix == '.pickle':
            wcl_o: WallCenterLineWithOpenPoints = load_wcl_o('../experiments/raw_demo/2_1k8_wcl.pickle')
            dict_ = wcl_o.json

            rooms = dict_.get('Rooms')
            for room in rooms:
                room['type'] = reverse_p.get(room.get('type'))

            save_json(dict_, path.parent / (path.name.rsplit('_', 1)[0] + '_serialized.json'))


if __name__ == '__main__':
    pass

