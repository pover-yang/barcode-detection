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

    # normalize to [0, 1]
    hmap_image = (hmap_image - hmap_image.min()) / (hmap_image.max() - hmap_image.min())
    blended_image = blend_hmap_img(image_tensor, hmap_tensor)

    # convert to heat-map
    hmap_image = cv2.cvtColor(hmap_image, cv2.COLOR_RGB2GRAY)
    # hmap_image = cv2.applyColorMap(np.uint8(255 * hmap_image), cv2.COLORMAP_JET)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('image')
    ax[1].imshow(hmap_image)
    ax[1].set_title('heat-map')
    ax[2].imshow(blended_image)
    ax[2].set_title('blended image')
    plt.show()


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
    blended_image = cv2.addWeighted(image, 0.1, hmap, 0.9, 0)

    return blended_image


if __name__ == '__main__':
    # batch_infer(
    #     model_path=r"D:\ExpLogs\UNet\v1\checkpoints\epoch=0-step=3714.ckpt",
    #     data_dir=r"D:\Barcode-Detection-Data"
    # )

    infer_single_image(
        model_path=r"D:\ExpLogs\UNet\v5\checkpoints\unet-epoch=049-val_loss=0.0014.ckpt",
        # model_path=r"D:\ExpLogs\UNet\v2\checkpoints\unet-epoch=087-val_loss=0.0010.ckpt",
        image_path=r"C:\Program Files (x86)\SMoreViScanner\capture\20230206015640270.png"
        # image_path=r"D:\Barcode-Detection-Data\data\0070\20210823044229002.png_jpg.png"
        # image_path=r"D:\Barcode-Detection-Data\data\0002\09-51-06_jpg.png"
    )
