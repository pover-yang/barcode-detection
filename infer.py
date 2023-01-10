from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from thop import profile
from torch.utils.data import DataLoader
from torchvision.utils import draw_bounding_boxes

from dataset.dataset import BarcodeCenterNetDataset, collate_fn
from lit_model import LitBarcodeDet
from models.centernet import CenterNet


def label_to_str(labels):
    label_map = {0: '1D', 1: 'QR', 2: 'DM'}
    label_strs: List[str] = []
    for label in labels:
        l_str = label_map[int(label)]
        label_strs.append(l_str)
    return label_strs


def infer_img(img_path, model_path):
    image = Image.open(img_path).convert('L')
    image_tensor = F.to_tensor(image)

    model = LitBarcodeDet.load_from_checkpoint(model_path)
    model.eval()

    model_out = model.model.forward([image_tensor])
    boxes = model_out[0]['boxes']
    labels = label_to_str(model_out[0]['labels'])
    font_size = min((box[3] - box[1]) // 2 for box in boxes)
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


def model_infer():
    # model = CenterNet().load_from_checkpoint('ckpt/epoch=112-step=42375.ckpt')
    model = CenterNet().load_from_checkpoint('ckpt/epoch=137-step=51750.ckpt')
    model.eval()

    dataset = BarcodeCenterNetDataset('/Users/yjunj/Data/Barcode-Detection-Data',
                                      'test')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    for images, targets in dataloader:
        model_out = model.forward(images)
        heat_map = model_out[0]
        heat_map[heat_map < 0.2] = 0
        blended_image = blend_hmap_img(images, heat_map)
        hmap_image = heat_map[0].detach().numpy().transpose((1, 2, 0))
        pass
    flops, params = profile(model, inputs=(images,))
    print(f"Flops: {flops / 1e9:.3f}G, Params: {params / 1e6:.3f}M")


def blend_hmap_img(image_tensor, hmap_tensor):
    image = image_tensor[0].detach().numpy().transpose((1, 2, 0))
    image = image.repeat(3, axis=2)
    hmap = hmap_tensor[0].detach().numpy().transpose((1, 2, 0))
    hmap = cv2.resize(hmap, (image.shape[1], image.shape[0]))
    blended_image = cv2.addWeighted(image, 0.3, hmap, 0.7, 0)

    return blended_image


if __name__ == '__main__':
    model_infer()
