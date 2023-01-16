import cv2
import json

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


def vertices_to_rot_rect(box_vertices):
    """
    Convert box points to rotated rectangle
    Args:
        box_vertices: Box points, shape (4, 2)

    Returns:
        rot_rect: Rotated rectangle, (center_x, center_y, width, height, angle)
    """

    # 1. Calculate the center of the box
    box_vertices = np.array(box_vertices)
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

    # 4. Convert to rotated rectangle
    rot_rect = ((x_center, y_center), (w, h), rot_angle)
    return rot_rect


def draw_boxes(img, boxes):
    """
    Draw boxes
    Args:
        img: Image
        boxes: List of boxes, shape (n, 4, 2)

    Returns:
        img: Image
    """
    for box in boxes:
        img = draw_box_vertices(img, box)
    return img


def draw_box_vertices(img, box_vertices, color=(0, 255, 0), thickness=2):
    """
    Draw box vertices
    Args:
        img: Image
        box_vertices: Box points, shape (4, 2)
        color: Color
        thickness: Thickness

    Returns:
        img: Image
    """
    for i in range(4):
        cv2.line(img, tuple(map(int, box_vertices[i])), tuple(map(int, box_vertices[(i + 1) % 4])),
                 (0, 255, 0), 2)
        cv2.circle(img, tuple(map(int, box_vertices[i])), 2, (0, 0, 255), 2)
        cv2.putText(img, str(i), tuple(map(int, box_vertices[i])), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

    return img


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
        heatmap[y_min:y_max, x_min:x_max] = np.maximum(heatmap[y_min:y_max, x_min:x_max], g)

    return heatmap


def main():
    image_path = '/Users/yjunj/Downloads/Image_20210416180119863_bmp.png'
    image_path = '/Users/yjunj/Data/Barcode-Detection-Data/data/0010/20210803154939745.png'
    image_path = '/Users/yjunj/Data/Barcode-Detection-Data/data/0004/Image_20210923102201175_bmp.png'
    json_path = image_path.replace('.png', '.json')

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    boxes = defaultdict(list)
    with open(json_path, 'r') as f:
        data = json.load(f)
        for shape in data['shapes']:
            if shape['label'] != '7':
                pt = shape['points'][0]
                group_id = shape['group_id']
                boxes[group_id].append(pt)

    rot_rects = []
    for box_vertices in boxes.values():
        rot_rect = vertices_to_rot_rect(box_vertices)
        rot_rects.append(rot_rect)

    heatmap = generate_elliptical_gaussian_heatmap(image.shape[:2], rot_rects)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image)
    ax[0].set_title('Image')
    ax[0].axis('off')

    ax[1].imshow(heatmap, cmap='jet')
    ax[1].set_title('Heatmap')
    ax[1].axis('off')
    plt.show()


if __name__ == '__main__':
    main()

