import json
from shutil import copy
from pathlib import Path
from collections import defaultdict


def statistics(dir_name):
    suffixes = set()
    num_files = 0
    files = list(dir_name.rglob("**/*.*"))
    for file in files:
        if file.is_file():
            segs = file.parts[-1].split('.')
            if len(segs) >= 2 and segs[-1].isalpha():
                suffixes.add(segs[-1])
                # if segs[-1] not in ('jpg', 'jpeg', 'png', 'bmp', 'JPG', 'JPEG', 'PNG', 'BMP'):
                if segs[-1] not in ('json'):
                    print(file)
                    copy(file, './unusual_files')
                    file.unlink()
                else:
                    num_files += 1
            else:
                print(file)
                file.unlink()
    print(suffixes)
    print(f"Num of files = {num_files}")


def find_label(dir_name, dir_type):
    files = list(dir_name.rglob(r"**\*.*"))
    num_wn_label = 0
    num_w_label = 0
    for file in files:
        if file.is_file():
            label_file = Path(str(file).replace(file.suffix, '.json').replace(f'\\{dir_type}', '\\label'))
            if not label_file.exists():
                print(label_file)
                num_wn_label += 1
            else:
                # 将file的复制到label文件夹下
                copy(file, str(label_file).replace(label_file.name, '').replace('\\label', '\\label-show'))
                num_w_label += 1
    print(num_wn_label)
    print(num_w_label)


def find_image(dir_name):
    files = list(dir_name.rglob(r"**\*.*"))
    for file in files:
        if file.is_file():
            for suffix in ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP'):
                train_image_file = Path(str(file).replace('.json', suffix).replace(f'\\label', '\\train'))
                test_image_file = Path(str(file).replace('.json', suffix).replace(f'\\label', '\\test'))
                if train_image_file.exists():
                    Path(str(file).replace(file.name, '').replace('\\label', '\\img-label')).mkdir(parents=True, exist_ok=True)
                    copy(file, str(file).replace(file.name, '').replace('\\label', '\\img-label'))
                    copy(train_image_file, str(train_image_file).replace(train_image_file.name, '').replace('\\train', '\\img-label'))
                if test_image_file.exists():
                    Path(str(file).replace(file.name, '').replace('\\label', '\\img-label')).mkdir(parents=True, exist_ok=True)
                    copy(file, str(file).replace(file.name, '').replace('\\label', '\\img-label'))
                    copy(test_image_file, str(test_image_file).replace(test_image_file.name, '').replace('\\test', '\\img-label'))


def label_type_statistics(dir_name):
    label_files = list(dir_name.rglob(r"**\*.json"))
    label_num_dict = defaultdict(int)

    point_label_dict = defaultdict(int)
    rect_label_dict = defaultdict(int)
    polygon_label_dict = defaultdict(int)

    for file in label_files:
        if file.is_file():
            content = json.load(open(file, mode='r', encoding='utf-8'))
            shapes = content["shapes"]
            for shape in shapes:
                label_type = shape["shape_type"]
                label_num_dict[label_type] += 1
                if label_type == 'point':
                    point_label_dict[shape["label"]] += 1
                elif label_type == 'rectangle':
                    rect_label_dict[shape["label"]] += 1
                elif label_type == 'polygon':
                    polygon_label_dict[shape["label"]] += 1
                    print(file)
    print(label_num_dict)
    print(point_label_dict)
    print(rect_label_dict)
    print(polygon_label_dict)