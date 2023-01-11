from typing import List

import cv2
from thop import profile
from torch.utils.data import DataLoader

from dataset.dataset_centernet import CenterNetDataset, collate_fn
from models.centernet import CenterNet


def model_infer(model_path, data_dir):
    model = CenterNet().load_from_checkpoint(model_path)
    model.eval()

    dataset = CenterNetDataset(data_dir, mode='test')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    for images, targets in dataloader:
        model_out = model.forward(images)
        images = images * 0.2349 + 0.4330
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
    blended_image = cv2.addWeighted(image, 0.2, hmap, 0.8, 0)

    return blended_image


if __name__ == '__main__':
    model_infer(model_path='ckpt/centernet-resnet18-epoch=21-val_loss=7.43.ckpt',
                data_dir='/Users/yjunj/Data/Barcode-Detection-Data')
