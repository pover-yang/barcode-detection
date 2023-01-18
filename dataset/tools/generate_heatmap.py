import cv2
import random
import tqdm
import numpy as np
from pathlib import Path

from multiprocessing import Process, Pool, cpu_count
from joblib import Parallel, delayed
from PIL import Image

from convert_label import parse_label_file, process_instances
from misc import blend_heatmap, split_train_test


def cal_oeg_hmap(img_size, n_classes, instances):
    """
    Generate **oriented elliptical** gaussian heatmap
    Args:
        img_size: Image size, (height, width)
        n_classes: Number of classes
        instances: List of instances, instance format: [x, y, w, h, theta, class_id]
    Returns:
        heatmap: Heatmap
    """
    heatmap = np.zeros((img_size[0], img_size[1], n_classes), dtype=np.float16)
    x_range = np.arange(0, img_size[1])
    y_range = np.arange(0, img_size[0])
    x_map, y_map = np.meshgrid(x_range, y_range)

    for instance in instances:
        rot_rect = (instance[:2], instance[2:4], instance[4]) # instance = rrect + [label]
        label = int(instance[5])

        # 0. The minium bounding box of the rotated rectangle
        box = cv2.boxPoints(rot_rect)
        box = np.intp(box)
        x_min, y_min = np.min(box, axis=0)
        x_max, y_max = np.max(box, axis=0)
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(img_size[1], x_max)
        y_max = min(img_size[0], y_max)

        # 1. The line function of the box's axes (a*x+b*y+c)
        x_center, y_center = rot_rect[0]
        rot_angle = rot_rect[2]

        a1 = np.cos(np.deg2rad(rot_angle))
        b1 = np.sin(np.deg2rad(rot_angle))
        c1 = -a1 * x_center - b1 * y_center
        const1 = np.sqrt(a1 ** 2 + b1 ** 2)

        a2 = np.cos(np.deg2rad(rot_angle + 90))
        b2 = np.sin(np.deg2rad(rot_angle + 90))
        c2 = -a2 * x_center - b2 * y_center
        const2 = np.sqrt(a2 ** 2 + b2 ** 2)

        # 2. Determine the sigma of the gaussian function by the width and height of the box
        w, h = rot_rect[1]
        sigma1 = w / 6
        sigma2 = h / 6

        # 3. Calculate the distance of each pixel to the box's axes line
        x = x_map[y_min:y_max, x_min:x_max]
        y = y_map[y_min:y_max, x_min:x_max]
        d1 = np.abs(a1 * x + b1 * y + c1) / const1
        g1 = np.exp(-d1 ** 2 / (2 * sigma1 ** 2))
        d2 = np.abs(a2 * x + b2 * y + c2) / const2
        g2 = np.exp(-d2 ** 2 / (2 * sigma2 ** 2))
        g = g1 * g2

        heatmap[y_min:y_max, x_min:x_max, label] = np.maximum(heatmap[y_min:y_max, x_min:x_max, label], g)

    return heatmap


def generate_heatmap_part(img_paths):
    """
    Generate oriented elliptical gaussian heatmap, and save it to the save_dir
    Args:
        img_paths: List of image paths
    """
    for img_path in tqdm.tqdm(img_paths):
        img = Image.open(img_path).convert("L")
        w, h = img.size[:2]

        json_path = img_path.with_suffix('.json')
        vertices_map, pids_map, img_w, img_h = parse_label_file(json_path)
        instances = process_instances(vertices_map, pids_map, dst_format='rrect')
        heatmap = cal_oeg_hmap((h, w), n_classes=3, instances=instances)

        # blended_img = blend_heatmap(img, heatmap)
        # vis_path = str(img_path).replace('data', 'visualize-v2')
        # cv2.imwrite(vis_path, blended_img)

        npy_path = str(img_path).replace('data', 'heatmap-v1').replace('.png', '.npy')
        np.save(npy_path, heatmap)


def generate_heatmap(root_dir):
    """
    Generate oriented elliptical gaussian heatmap, and save it to the save_dir
    Args:
        root_dir: Root directory
    """
    # split the img_paths into 8 parts, and generate heatmap in parallel
    img_paths = list(root_dir.rglob('*.png'))

    img_paths_parts = np.array_split(img_paths, 16)
    img_paths_parts = [list(i) for i in img_paths_parts]

    # multiprocess to generate heatmap
    # n_core = cpu_count()
    # with Pool(n_core) as p:
    #     p.map(generate_heatmap_part, img_paths_parts)
    n_core = cpu_count()
    Parallel(n_jobs=n_core)\
        (delayed(generate_heatmap_part)(img_paths_part) for img_paths_part in img_paths_parts)


def generate_files_indices(root_dir, train_ratio=0.8):
    root_dir = Path(root_dir)
    all_indices_file = root_dir.parent / 'hmap_all.txt'
    train_indices_file = root_dir.parent / 'hmap_train.txt'
    test_indices_file = root_dir.parent / 'hmap_test.txt'

    img_paths = list(root_dir.rglob('*.png'))
    img_paths = [str(i) + '\n' for i in img_paths]
    random.shuffle(img_paths)

    train_indices = img_paths[:int(len(img_paths) * train_ratio)]
    test_indices = img_paths[int(len(img_paths) * train_ratio):]

    with open(all_indices_file, 'w') as f:
        f.writelines(img_paths)

    with open(train_indices_file, 'w') as f:
        f.writelines(train_indices)

    with open(test_indices_file, 'w') as f:
        f.writelines(test_indices)


def main():
    root_dir = Path(r"D:\Barcode-Detection-Data\data")
    generate_files_indices(root_dir)

    # heatmap_dir = Path(root_dir).parent / 'heatmap-v1'
    # heatmap_dir.mkdir(exist_ok=True)
    # # visualize_dir = Path(root_dir).parent / 'visualize-v2'
    # # visualize_dir.mkdir(exist_ok=True)
    # for sub_dir in root_dir.iterdir():
    #     sub_save_dir = heatmap_dir / sub_dir.name
    #     sub_save_dir.mkdir(exist_ok=True)
    #
    #     # sub_vis_dir = visualize_dir / sub_dir.name
    #     # sub_vis_dir.mkdir(exist_ok=True)
    #
    # generate_heatmap(root_dir)



if __name__ == '__main__':
    main()

