import cv2
from thop import profile
from torch.utils.data import DataLoader

from dataset.heatmap_ds import HeatMapDataset
from models import LitUNet
from configs.unet_cf_win import conf


def model_infer(model_path, data_dir):
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
    blended_image = cv2.addWeighted(image, 0.2, hmap, 0.8, 0)

    return blended_image


if __name__ == '__main__':
    model_infer(
        model_path=r"D:\ExpLogs\UNet\v1\checkpoints\epoch=0-step=3714.ckpt",
        data_dir=r"D:\Barcode-Detection-Data"
    )
