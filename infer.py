from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from numpy import ndarray
from torch import Tensor
from torchvision.utils import draw_bounding_boxes

from lit_model import LitBarcodeDet


def label_to_str(labels: Tensor) -> List[str]:
    label_map = {0: '1D', 1: 'QR', 2: 'DM'}
    label_strs: List[str] = []
    for label in labels:
        l_str = label_map[int(label)]
        label_strs.append(l_str)
    return label_strs


def infer_img(img_path: str, model_path: str) -> ndarray:
    image = Image.open(img_path).convert('L')
    image_tensor = F.to_tensor(image)

    model = LitBarcodeDet.load_from_checkpoint(model_path)
    model.eval()

    model_out = model.model.forward([image_tensor])
    boxes = model_out[0]['boxes']
    labels = label_to_str(model_out[0]['labels'])
    font_size = min((box[3]-box[1]) // 2 for box in boxes)
    out_image = draw_bounding_boxes((image_tensor * 255).to(torch.uint8),
                                    boxes=boxes,
                                    labels=labels,
                                    colors="red",
                                    fill=True,
                                    font='/System/Library/Fonts/Supplemental/Andale Mono.ttf',
                                    font_size=int(font_size))
    vis_image = np.asarray(out_image).transpose((1, 2, 0))
    return vis_image


def main():
    img_path = '/Users/yjunj/Downloads/10bit_tm.png'

    model_path = '/Users/yjunj/Downloads/barcode-det-epoch=050.ckpt'
    vis_img = infer_img(img_path, model_path)
    plt.imshow(vis_img)
    plt.show()


if __name__ == '__main__':
    main()
