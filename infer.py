import cv2
import numpy as np
import torch
from thop import profile
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

from dataset.heatmap_ds import HeatMapDataset
from models import LitUNet
from configs.unet_cf_win import conf
import matplotlib.pyplot as plt


# Inference on a single image
def infer_single_image(model_path, image_path):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    model = LitUNet(conf.model).load_from_checkpoint(model_path, model_conf=conf.model)
    model.eval()

    # preprocess
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_tensor = preprocess(image)
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('input image')

    # inference
    hmap_tensor = model(image_tensor)

    # postprocess
    hmap = postprocess(hmap_tensor)
    ax[1].imshow(hmap)
    ax[1].set_title('heat map')

    # blend
    image = (image / 255.0).astype(np.float32)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    cv2.addWeighted(image, 0.5, hmap, 0.5, 0, hmap)
    ax[2].imshow(hmap)
    ax[2].set_title('blended image')
    plt.show()


def preprocess(image):
    image = (image / 255.0).astype(np.float32)
    image = (image - 0.4330) / 0.2349
    image = cv2.resize(image, (640, 400))
    image_tensor = TF.to_tensor(image).unsqueeze(0)
    return image_tensor


def postprocess(hmap):
    hmap = hmap[0].detach().numpy()
    hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min())
    hmap = cv2.resize(hmap, (1280, 800))
    return hmap


def batch_infer(model_path, data_dir):
    model = LitUNet(conf.model).load_from_checkpoint(model_path, model_conf=conf.model)
    model.eval()

    dataset = HeatMapDataset(data_dir, mode='test')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for images, targets in dataloader:
        model_out = model.forward(images)
        images = images * 0.2349 + 0.4330
        heat_map = model_out
        pass
    flops, params = profile(model, inputs=(images,))
    print(f"Flops: {flops / 1e9:.3f}G, Params: {params / 1e6:.3f}M")


def to_onnx(model_path, onnx_path):
    model = LitUNet(conf.model).load_from_checkpoint(model_path, model_conf=conf.model)
    model.eval()

    dummy_input = torch.randn(1, 1, 400, 640)
    torch.onnx.export(model, dummy_input, onnx_path, verbose=True, opset_version=11)


if __name__ == '__main__':
    to_onnx("ckpt/hmap-v3-e99-fp32.ckpt",
            "ckpt/hmap-v3-e99-fp32.onnx")

    # infer_single_image(
    #     model_path="ckpt/hmap-v3-e99-fp32.ckpt",
    #     image_path="data/test1.png"
    # )
