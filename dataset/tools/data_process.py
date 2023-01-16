import json
from pathlib import Path
from collections import defaultdict
from PIL import Image
import numpy as np
import tqdm


valid_pids_map = {
    (0, 0, 0, 0): 0,  # 1d
    (8, 9, 10, 11): 0,  # 1d
    (1, 1, 2, 3): 1,  # qr
    (1, 1, 2, 3, 7, 7, 7): 1,  # qr
    (4, 4, 5, 6): 2  # dm
}


# 将四个点坐标转换为bbox坐标
def vertices_to_bbox(vertices):
    x_min, y_min = np.min(vertices, axis=0)
    x_max, y_max = np.max(vertices, axis=0)
    return [x_min, y_min, x_max, y_max]


def vertices_to_rot_rect(vertices):
    """
    Convert box points to rotated rectangle
    Args:
        vertices: Box points, shape (4, 2)

    Returns:
        # rot_rect: Rotated rectangle, ((center_x, center_y), (width, height), angle)
        rot_rect: Rotated rectangle, x_center, y_center, w, h, rot_angle
    """

    # 1. Calculate the center of the box
    box_vertices = np.array(vertices)
    x_center, y_center = np.mean(box_vertices, axis=0)

    # 2. Calculate the rotation angle of the box
    y_s = box_vertices[:, 1] - y_center
    x_s = box_vertices[:, 0] - x_center
    angles = np.arctan2(y_s, x_s)
    rot_angle = np.rad2deg(np.mean(angles))

    # 3. Calculate the width and height of the box(Sort the points according to the angle first)
    sorted_idx = np.argsort(angles)
    box_vertices = box_vertices[sorted_idx]

    w = np.linalg.norm(box_vertices[1] - box_vertices[0])
    h = np.linalg.norm(box_vertices[3] - box_vertices[0])

    # # 4. Convert to rotated rectangle
    # rot_rect = ((x_center, y_center), (w, h), rot_angle)
    return x_center, y_center, w, h, rot_angle


# 将所有图像对应的标签文件转换为一个文件， 第一列为图像文件名，后面为bbox的坐标以及类别
def label_process_to_bbox(dir_name):
    dst_txt_file = dir_name.parent / 'all_bbox.txt'
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
                group_bbox = vertices_to_bbox(group_coord)

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


# 将所有图像对应的标签文件转换为一个文件， 第一列为图像文件名，后面为rotated rectangle以及类别
def label_process_to_rrect(dir_name):
    dst_txt_file = dir_name.parent / 'all_rrect.txt'
    f = open(dst_txt_file, 'w')

    # 读取所有的标签文件
    for label_path in list(dir_name.rglob(r"**/*.json")):
        relative_path = label_path.with_suffix('.png').relative_to(dir_name)
        label_line = [str(relative_path)]

        json_obj = json.load(open(label_path, mode='r', encoding='utf-8'))
        img_w, img_h = json_obj['imageWidth'], json_obj['imageHeight']

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
                group_rect = vertices_to_rot_rect(group_coord)

                # 过滤掉rect长宽不足图像长宽1/100的bbox
                rect_w, rect_h = group_rect[1]
                if rect_w / img_w < 0.01 or rect_w / img_w < 0.01:
                    print(f"Invalid rect size: {(rect_w, rect_h)} in {label_path}")
                    continue

                group_rect.append(group_label)
                bbox_label_str = ",".join(str(n) for n in group_rect)
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


# 统计灰度图像数据集的均值方差
def cal_mean_std(dir_name):

    mean = np.zeros(1)
    std = np.zeros(1)
    all_img_paths = list(dir_name.rglob(r"**/*.png"))
    for img_path in tqdm.tqdm(all_img_paths, ncols=80):
        pil_img = Image.open(img_path).convert("L")
        img = np.array(pil_img, dtype=np.float64)
        img /= 255.0
        mean += img.mean()
    mean /= len(all_img_paths)

    for img_path in tqdm.tqdm(all_img_paths, ncols=80):
        pil_img = Image.open(img_path).convert("L")
        img = np.array(pil_img, dtype=np.float64)
        img /= 255.0
        diff = (img - mean).mean()
        std += diff * diff
    std /= len(all_img_paths)
    std = np.sqrt(std)

    print(f"mean: {mean}, std: {std}")
    with open(dir_name.parent / 'mean_std.txt', 'w', encoding='utf-8') as f:
        f.writelines(f"mean: {mean}, std: {std}")


def main():
    label_process_to_rrect(Path(r'D:\Barcode-Detection-Data\data'))
    split_train_test_txt(Path('/home/junjieyang/Data/Barcode-Detection-Data/all.txt'))


if __name__ == "__main__":
    main()
