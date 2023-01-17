import cv2
import numpy as np


def generate_elliptical_gaussian_heatmap(img_size, rot_rects):
    """
    Generate oriented elliptical gaussian heatmap
    Args:
        img_size: Image size, (height, width)
        rot_rects: List of  Rotated rectangle, (center_x, center_y, width, height, angle)

    Returns:
        heatmap: Heatmap
    """
    x_range = np.arange(0, img_size[1])
    y_range = np.arange(0, img_size[0])
    x_map, y_map = np.meshgrid(x_range, y_range)
    heatmap = np.zeros(img_size, dtype=np.float32)

    for rot_rect in rot_rects:
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
        heatmap[y_min:y_max, x_min:x_max] = np.maximum(heatmap[y_min:y_max, x_min:x_max], g)

    return heatmap
