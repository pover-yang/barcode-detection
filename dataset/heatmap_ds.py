from pathlib import Path

import torch
import numpy as np
import torchvision.transforms.functional as F

from PIL import Image
from torch.utils.data import Dataset, DataLoader

from dataset.centernet_ts import CenterNetTransform


class HeatMapDataset(Dataset):
    def __init__(self, root_dir, mode, input_size=1024):
        self.img_paths = list((Path(root_dir) / 'data').rglob('*.png'))
        self.hmap_paths = list((Path(root_dir) / 'hmap').rglob('*.npy'))

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("L")
        image = F.to_tensor(image)

        hmap_path = self.hmap_paths[idx]
        hmap = torch.from_numpy(np.load(str(hmap_path)))

        return image, hmap

    def __len__(self):
        return len(self.img_paths)


def collate_fn(batch):
    imgs = []
    hmaps, whs, offsets, reg_masks = [], [], [], []
    for img, target in batch:
        imgs.append(img)
        if target is not None:
            hmap, wh, offset, reg_mask = target.values()
            hmaps.append(hmap)
            whs.append(wh)
            offsets.append(offset)
            reg_masks.append(reg_mask)

    imgs = torch.stack(imgs, dim=0)
    if len(hmaps) > 0:
        targets = {'hmap': torch.stack(hmaps, dim=0),
                   'wh': torch.stack(whs, dim=0),
                   'offset': torch.stack(offsets, dim=0),
                   'reg_mask': torch.stack(reg_masks, dim=0)}
    else:
        targets = None

    return imgs, targets
