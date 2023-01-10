import json
from pathlib import Path
from collections import defaultdict

valid_pids_set = [[0, 0, 0, 0],  # 1d
                  [8, 9, 10, 11],  # 1d
                  [1, 1, 2, 3],  # qr
                  [1, 1, 2, 3, 7, 7, 7],  # qr
                  [4, 4, 5, 6]]  # dm

valid_pids_map = {
    (0, 0, 0, 0): 0,  # 1d
    (8, 9, 10, 11): 0,  # 1d
    (1, 1, 2, 3): 1,  # qr
    (1, 1, 2, 3, 7, 7, 7): 1,  # qr
    (4, 4, 5, 6): 2  # dm
}

# 将四个点坐标转换为bbox坐标
def coords_to_bbox(coords):
    x_coords = [coord[0] for coord in coords]
    y_coords = [coord[1] for coord in coords]

    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)

    return [x_min, y_min, x_max, y_max]


# 将所有图像对应的标签文件转换为一个文件， 第一列为图像文件名，后面为bbox的坐标以及类别
def label_process(dir_name):
    dst_txt_file = dir_name.parent / 'all.txt'
    f = open(dst_txt_file, 'w')

    # 读取所有的标签文件
    for label_path in list(dir_name.rglob(r"**/*.json")):
        relative_path = label_path.with_suffix('.png').relative_to(dir_name)
        label_line = [str(relative_path)]

        json_obj = json.load(open(label_path, mode='r', encoding='utf-8'))
        img_size = json_obj['imageHeight'], json_obj['imageWidth']

        group_coords = defaultdict(list)
        group_pids = defaultdict(list)

        for shape in json_obj['shapes']:
            if shape['shape_type'] != 'point' or int(shape['label']) == -1:
                continue

            group_id = shape['group_id'] if shape['group_id'] else 0
            p_coord = shape['points'][0]
            pid = int(shape['label'])

            group_coords[group_id].append(p_coord)
            group_pids[group_id].append(pid)

        group_bboxes = []
        # 检查每个group的pid组合是否在有效组合中
        for group_id, pids in group_pids.items():
            pids.sort()
            if tuple(pids) not in valid_pids_map.keys():
                print(f"Invalid pids: {pids} in {label_path}")
            else:
                group_label = valid_pids_map[tuple(pids)]
                group_coord = group_coords[group_id]
                group_bbox = coords_to_bbox(group_coord)

                # 过滤掉bbox长宽不足图像长宽1/100的bbox
                bbox_size = group_bbox[2] - group_bbox[0], group_bbox[3] - group_bbox[1]
                if bbox_size[0] / img_size[0] < 0.01 or bbox_size[1] / img_size[1] < 0.01:
                    print(f"Invalid bbox size: {bbox_size} in {label_path}")
                    continue

                group_bbox.append(group_label)
                bbox_label_str = ",".join(str(n) for n in group_bbox)
                label_line.append(bbox_label_str)

        if len(label_line) > 1:
            f.writelines("\t".join(label_line)+"\n")

    f.close()


# 将所有数据记录的txt随机分为训练集和测试集
def split_train_test_txt(txt_file, train_ratio=0.8):
    with open(txt_file, 'r') as f:
        lines = f.readlines()

    import random
    random.shuffle(lines)

    train_lines = lines[:int(len(lines) * train_ratio)]
    test_lines = lines[int(len(lines) * train_ratio):]

    with open(txt_file.parent / 'train.txt', 'w', encoding='utf-8') as f:
        f.writelines(train_lines)

    with open(txt_file.parent / 'test.txt', 'w') as f:
        f.writelines(test_lines)


def main():
    label_process(Path('/home/junjieyang/Data/Barcode-Detection-Data/data/'))
    split_train_test_txt(Path('/home/junjieyang/Data/Barcode-Detection-Data/all.txt'))


if __name__ == "__main__":
    main()
