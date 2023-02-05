import torch
from pathlib import Path
from torchvision.utils import make_grid


def next_version(log_root, exp_name):
    exp_dirs = [d for d in (Path(log_root) / exp_name).iterdir() if d.is_dir()]
    version_nums = []
    for d in exp_dirs:
        if d.name.startswith('v') and d.name[1:].isdigit():
            version_nums.append(int(d.name[1:]))

    return max(version_nums) + 1 if version_nums else 1


def blend_image_hmap_tensor(img, hmap, alpha=0.5):
    img = img * 0.2349 + 0.4330
    hmap = torch.sigmoid(hmap)
    blended_batch = img * alpha + hmap * (1 - alpha)
    blended_batch = (blended_batch - blended_batch.min()) / (blended_batch.max() - blended_batch.min())
    blended_grid = make_grid(blended_batch)
    return blended_grid
