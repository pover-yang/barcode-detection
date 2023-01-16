import cv2
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F


class CenterNetTransform(nn.Module):

    def __init__(self, input_size, output_size, image_mean, image_std, num_classes, min_overlap=0.7):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.num_classes = num_classes
        self.min_overlap = min_overlap

    def forward(self, image, target):
        image, target = self.resize(image, target)
        target = self.generate_target(target)

        image, target = self.to_tensor(image, target)
        image = self.normalize(image)
        return image, target

    def resize(self, image, target=None):
        img_w, img_h = image.size
        dst_w, dst_h = self.input_size
        scale = min(dst_w / img_w, dst_h / img_h)

        scaled_img_w, scaled_img_h = (int(img_w * scale), int(img_h * scale))
        dx = (dst_w - scaled_img_w) // 2
        dy = (dst_h - scaled_img_h) // 2
        # scaled_image = F.resize(image, [scaled_img_h, scaled_img_w], interpolation=F.InterpolationMode.BILINEAR)
        scaled_image = image.resize((scaled_img_w, scaled_img_h), Image.Resampling.BILINEAR)
        dst_img = Image.new('L', self.input_size, 128)
        dst_img.paste(scaled_image, (dx, dy))  # paste image to center

        if target is not None:
            dst_target = []
            for instance in target:
                x1, y1, x2, y2, label = instance
                # 坐标限制在图片内(防止resize后超出图片边界)
                x1 = min(max(0, x1 * scale + dx), dst_w - 1)
                y1 = min(max(0, y1 * scale + dy), dst_h - 1)
                x2 = min(max(0, x2 * scale + dx), dst_w - 1)
                y2 = min(max(0, y2 * scale + dy), dst_h - 1)
                label = int(label)
                dst_target.append([x1, y1, x2, y2, label])
        else:
            dst_target = None
        return dst_img, dst_target

    def normalize(self, image):
        image = (image - self.image_mean) / self.image_std
        return image

    @staticmethod
    def generate_heatmap(img_size, num_class, rot_rects, labels):
        """
        Generate oriented elliptical gaussian heatmap
        Args:
            img_size: Image size, (height, width)
            num_class: Number of classes
            rot_rects: List of  Rotated rectangle, (center_x, center_y, width, height, angle)
            labels: Class label of each rotated rectangle

        Returns:

        """
        x_range = np.arange(0, img_size[1])
        y_range = np.arange(0, img_size[0])
        x_map, y_map = np.meshgrid(x_range, y_range)
        hmap = np.zeros((img_size[0], img_size[1], num_class), dtype=np.float32)

        for rot_rect, label in zip(rot_rects, labels):
            # 0. The minium bounding box of the rotated rectangle
            box = cv2.boxPoints(rot_rect)
            box = np.int0(box)
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
            g_d1 = np.exp(-d1 ** 2 / (2 * sigma1 ** 2))
            d2 = np.abs(a2 * x + b2 * y + c2) / const2
            g_d2 = np.exp(-d2 ** 2 / (2 * sigma2 ** 2))
            g = g_d1 * g_d2

            hmap[y_min:y_max, x_min:x_max, label] = np.maximum(hmap[y_min:y_max, x_min:x_max, label], g)
        return hmap

    @staticmethod
    def to_tensor(image, target=None):
        image = F.to_tensor(image)
        if target is not None:
            target = {k: F.to_tensor(v) for k, v in target.items()}
        else:
            target = None
        return image, target


def gaussian_radius(height, width, min_overlap=0.7):
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 - sq1) / (2 * a1)

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 - sq2) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / (2 * a3)

    radius = min(r1, r2, r3)
    # radius = max(r1, r2, r3)
    print(radius, height, width)
    return radius


def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1  # diameter of gaussian
    gaussian = gaussian2d((diameter, diameter), sigma=diameter / 6)

    h, w = heatmap.shape[:2]
    x, y = center

    # boundary of gaussian
    left, right = min(x, radius), min(w - x, radius + 1)
    top, bottom = min(y, radius), min(h - y, radius + 1)

    # select the region of interest
    masked_heatmap = heatmap[y - top: y + bottom, x - left: x + right]

    # limit the gaussian to the region of interest
    # masked_gaussian = gaussian[radius - top: radius + bottom, radius - left:radius + right]
    masked_gaussian = gaussian

    # 将高斯分布覆盖到heatmap上，相当于不断的在heatmap基础上添加关键点的高斯，
    # 即同一种类型的框会在一个heatmap某一个类别通道上面上面不断添加。
    # 最终通过函数总体的for循环，相当于不断将目标画到heatmap
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    return heatmap


def gaussian2d(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    # 限制最小的值
    return h
