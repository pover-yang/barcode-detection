import cv2
import numpy as np
import torch
from thop import profile
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

from dataset.heatmap_ds import HeatMapDataset
from models import LitUNet
from configs.unet_cf_win import conf


# Inference on a single image
def infer_single_image(model_path, image_path):
    model = LitUNet(conf.model).load_from_checkpoint(model_path, model_conf=conf.model)
    model.eval()

    # imread with grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (640, 400))
    image_norm = (image / 255.0).astype(np.float32)
    image_norm = (image_norm - 0.4330) / 0.2349

    image_tensor = TF.to_tensor(image_norm).unsqueeze(0)
    hmap_tensor = model(image_tensor)
    hmap_tensor = torch.sigmoid(hmap_tensor)
    hmap_image = hmap_tensor[0].detach().numpy().transpose((1, 2, 0))
    hmap_image = cv2.cvtColor(hmap_image, cv2.COLOR_BGR2RGB)
    blended_image = blend_hmap_img(image_tensor, hmap_tensor)
    cv2.imshow('image with heat-map', blended_image)
    cv2.imshow('heat-map', hmap_image)
    cv2.imshow('image', image)
    cv2.waitKey(0)


def batch_infer(model_path, data_dir):
    model = LitUNet(conf.model).load_from_checkpoint(model_path, model_conf=conf.model)
    model.eval()

    dataset = HeatMapDataset(data_dir, mode='test')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for images, targets in dataloader:
        model_out = model.forward(images)
        images = images * 0.2349 + 0.4330
        heat_map = model_out
        blended_image = blend_hmap_img(images, heat_map)
        pass
    flops, params = profile(model, inputs=(images,))
    print(f"Flops: {flops / 1e9:.3f}G, Params: {params / 1e6:.3f}M")


def blend_hmap_img(image_tensor, hmap_tensor):
    image = image_tensor[0].detach().numpy().transpose((1, 2, 0))
    image = image.repeat(3, axis=2)
    hmap = hmap_tensor[0].detach().numpy().transpose((1, 2, 0))
    hmap = cv2.resize(hmap, (image.shape[1], image.shape[0]))
    blended_image = cv2.addWeighted(image, 0.5, hmap, 0.5, 0)

    return blended_image


if __name__ == '__main__':
    # batch_infer(
    #     model_path=r"D:\ExpLogs\UNet\v1\checkpoints\epoch=0-step=3714.ckpt",
    #     data_dir=r"D:\Barcode-Detection-Data"
    # )

    infer_single_image(
        model_path=r"D:\ExpLogs\UNet\remote-v2\checkpoints\unet-epoch=070-val_loss=0.0010.ckpt",
        image_path=r"C:\Program Files (x86)\SMoreViScanner\capture\20230206015640270.png"
    )
