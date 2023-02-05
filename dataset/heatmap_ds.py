from pathlib import Path

import cv2
import random
import torch
import numpy as np
import torchvision.transforms.functional as TF
from torchvision.transforms import *

from PIL import Image
from torch.utils.data import Dataset, DataLoader


class HeatMapDataset(Dataset):
    def __init__(self, root_dir, mode, input_size=(512, 512)):
        self.eval = (mode != 'train')
        self.img_dir = Path(root_dir) / 'data'
        self.label_dir = Path(root_dir) / 'rrect'
        self.img_paths, self.list_instances = [], []

        self.input_size = input_size
        self.load_items(mode)

        self.color_jitter = ColorJitter(brightness=0.01, contrast=0, saturation=0, hue=0)

    def load_items(self, mode):
        sample_file = self.label_dir / f'{mode}.txt'

        sample_items = open(sample_file, mode='r', encoding='utf-8').readlines()
        for item in sample_items:
            item_parts = item.strip().split('\t')
            img_path = self.img_dir / item_parts[0]
            instances = [list(map(float, instance.split(','))) for instance in item_parts[1:]]

            self.img_paths.append(img_path)
            self.list_instances.append(instances)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("L")

        instances = self.list_instances[idx]
        hmap = self.cal_oeg_hmap(image.size, n_classes=3, instances=instances)

        image_tensor, hmap_tensor = self.transforms(image, hmap)
        return image_tensor, hmap_tensor

    def __len__(self):
        return len(self.img_paths)

    def transforms(self, image, hmap):
        # ToTensor
        image = TF.to_tensor(image)
        hmap = TF.to_tensor(hmap)

        # Random adjust image's brightness, contrast, blur
        if random.random() < 0.3 and not self.eval:
            factor = random.uniform(0.5, 2)
            image = TF.adjust_brightness(image, factor)

        if random.random() < 0.3 and not self.eval:
            factor = random.uniform(0.5, 2)
            image = TF.adjust_contrast(image, factor)

        if random.random() < 0.3 and not self.eval:
            ksize = random.choice((3, 5, 7, 9))
            image = TF.gaussian_blur(image, kernel_size=[ksize, ksize])

        # Normalize
        image = TF.normalize(image, mean=[0.4330], std=[0.2349])

        # Random horizontal flip
        if random.random() < 0.5 and not self.eval:
            image = TF.hflip(image)
            hmap = TF.hflip(hmap)

        # Random vertical flip
        if random.random() < 0.5 and not self.eval:
            image = TF.vflip(image)
            hmap = TF.vflip(hmap)

        # Random rotation
        if random.random() < 0.5 and not self.eval:
            angle = random.randint(-90, 90)
            image = TF.rotate(image, angle)
            hmap = TF.rotate(hmap, angle)

        # Random crop resize
        if random.random() < 0.7 and not self.eval:
            image, hmap = self.random_crop_resize(image, hmap)
        else:
            image, hmap = self.resize_keep_aspect(image, hmap)

        return image, hmap

    def random_crop_resize(self, image, hmap, crop_prob=0.2):
        # random crop
        if random.random() < crop_prob:
            img_size = list(image.shape[-2:])

            crop_rate = random.uniform(0.6, 0.8)
            crop_size = [int(img_size[0] * crop_rate), int(img_size[1] * crop_rate)]

            offset_rate = random.uniform(-0.2, 0.2)
            dy, dx = [int(img_size[0] * offset_rate), int(img_size[1] * offset_rate)]

            image = TF.crop(image, dy, dx, crop_size[0], crop_size[1])
            hmap = TF.crop(hmap, dy, dx, crop_size[0], crop_size[1])
        # resize
        image = TF.resize(image, [self.input_size[0], self.input_size[1]])
        hmap = TF.resize(hmap, [self.input_size[0], self.input_size[1]])
        return image, hmap

    def resize_keep_aspect(self, image, hmap):
        dst_image = torch.full((1, self.input_size[0], self.input_size[1]), 0.5, dtype=torch.float32)
        dst_hmap = torch.zeros((3, self.input_size[0], self.input_size[1]), dtype=torch.float32)

        img_h, img_w = image.shape[-2:]
        scale = min(self.input_size[0] / img_h, self.input_size[1] / img_w)
        scaled_w, scaled_h = int(img_w * scale), int(img_h * scale)
        dx = (self.input_size[1] - scaled_w) // 2
        dy = (self.input_size[0] - scaled_h) // 2
        scaled_image = TF.resize(image, [scaled_h, scaled_w])
        scaled_hmap = TF.resize(hmap, [scaled_h, scaled_w])

        dst_image[:, dy:dy + scaled_h, dx:dx + scaled_w] = scaled_image
        dst_hmap[:, dy:dy + scaled_h, dx:dx + scaled_w] = scaled_hmap

        return dst_image, dst_hmap

    @staticmethod
    def cal_oeg_hmap(img_size, n_classes, instances):
        """
        Generate **oriented elliptical** gaussian heatmap
        Args:
            img_size: Image size, (width, height)
            n_classes: Number of classes
            instances: List of instances, instance format: [x, y, w, h, theta, class_id]
        Returns:
            heatmap: Heatmap
        """
        heatmap = np.zeros((img_size[1], img_size[0], n_classes), dtype=np.float32)
        x_range = np.arange(0, img_size[0])
        y_range = np.arange(0, img_size[1])
        x_map, y_map = np.meshgrid(x_range, y_range)

        for instance in instances:
            rot_rect = (instance[:2], instance[2:4], instance[4])  # instance = rrect + [label]
            label = int(instance[5])

            # 0. The minium bounding box of the rotated rectangle
            box = cv2.boxPoints(rot_rect)
            box = np.intp(box)
            x_min, y_min = np.min(box, axis=0)
            x_max, y_max = np.max(box, axis=0)
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(img_size[0], x_max)
            y_max = min(img_size[1], y_max)

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


def train_val_loader(conf):
    train_dataset = HeatMapDataset(conf.root_dir, mode='train')
    val_dataset = HeatMapDataset(conf.root_dir, mode='test')

    train_loader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True,
                              num_workers=conf.num_workers,
                              pin_memory=True,
                              # persistent_workers=True
                              )
    val_loader = DataLoader(val_dataset, batch_size=conf.batch_size, shuffle=False,
                            num_workers=conf.num_workers,
                            pin_memory=True,
                            # persistent_workers=True
                            )
    return train_loader, val_loader


def sample_loader(conf):
    sample_dataset = HeatMapDataset(conf.root_dir, mode='sample')
    sample_dataloader = DataLoader(sample_dataset, batch_size=4, shuffle=False, pin_memory=True)
    return sample_dataloader


if __name__ == '__main__':
    import torchvision
    dataset = HeatMapDataset(r'D:\Barcode-Detection-Data', mode='train', input_size=512)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    from utils.misc import blend_image_hmap_tensor
    for i, (im, hm) in enumerate(dataloader):
        blended_grid = blend_image_hmap_tensor(im, hm, alpha=0.3)
        blended_grid_np = blended_grid.numpy().transpose(1, 2, 0)
        cv2.imshow('blended', blended_grid_np)
        cv2.waitKey(0)
