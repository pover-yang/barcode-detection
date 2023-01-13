import json
from shutil import move
from pathlib import Path


# 查找存在对应json文件的图像，将图像和json文件复制到一个新的文件夹下
def copy_image_label(src_dir):
    mode = src_dir.name

    for img_path in src_dir.rglob(r"**\*.*"):
        if img_path.is_file() and img_path.suffix in ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP'):
            label_path = Path(str(img_path.with_suffix('.json')).replace(f'\\{mode}', '\\label'))
            if label_path.exists():
                dst_dir = Path(str(img_path.parent).replace(mode, 'data-copy'))
                dst_dir.mkdir(parents=True, exist_ok=True)

                move(img_path, dst_dir / img_path.name)
                move(label_path, dst_dir / label_path.name)
    print(f'copy {mode} done')


# 合并子文件夹下所有文件到一个文件夹下，重新编号
def merge_dir(root_dir):
    dst_dir = root_dir.parent / 'merge-dir'

    for sud_dir in root_dir.glob("*/"):
        dir_id = sud_dir.name.split('_')[1]
        dst_sub_dir = dst_dir / dir_id
        dst_sub_dir.mkdir(parents=True, exist_ok=True)

        for path in sud_dir.rglob("**/*.*"):
            if path.is_file():
                if (dst_sub_dir / path.name) in dst_sub_dir.glob("*"):
                    new_file_name = path.name.replace(path.suffix, f'_{path.parts[-2]}{path.suffix}')
                    move(path, dst_sub_dir / new_file_name)
                else:
                    move(path, dst_sub_dir / path.name)
    print('merge done')


def convert_image_format(dir_name):
    for label_path in dir_name.rglob(r"**\*.json"):
        json_obj = json.load(open(label_path, mode='r', encoding='utf-8'))
        img_name = json_obj['imagePath']
        json_img_path = label_path.parent / img_name
        img_path = json_img_path

        if not json_img_path.exists():
            print(f'{json_img_path} not exists')
            for suffix in ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP'):
                if label_path.with_suffix(suffix).exists():
                    img_path = label_path.with_suffix(suffix)

                    json_obj['imagePath'] = img_path.name
                    json.dump(json_obj, open(label_path, mode='w', encoding='utf-8'), indent=4, ensure_ascii=False)

                    print(f'change {json_img_path} to {img_path}')
                    break

        if img_path.suffix != '.png':
            new_img_path = Path(str(img_path).replace(img_path.suffix, f'_{img_path.suffix[1:]}.png'))
            new_label_file = new_img_path.with_suffix('.json')
            img_path.rename(new_img_path)
            label_path.rename(new_label_file)

            json_obj['imagePath'] = new_img_path.name
            json.dump(json_obj, open(new_label_file, mode='w', encoding='utf-8'), indent=4, ensure_ascii=False)

    print('convert done')


def main():
    # copy_image_label(Path(r'D:\original-data\train'))
    # copy_image_label(Path(r'D:\original-data\test'))
    # merge_dir(Path(r'D:\original-data\data-copy'))
    convert_image_format(Path(r"D:\original-data\merge-dir"))


if __name__ == "__main__":
    main()
