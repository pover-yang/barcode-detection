from pathlib import Path

import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset


class TorchvisionDataset(Dataset):
    def __init__(self, root_dir, sample_file):
        self.root_dir = Path(root_dir)
        self.img_paths, self.targets = [], []
        self.load_items(sample_file)

    # 用torchvision的格式从sample_file中读取图片路径和标签(bbox+label)
    def load_items(self, sample_file):
        item_lines = open(sample_file, 'r').readlines()
        for item_line in item_lines:
            line_parts = item_line.strip().split('\t')

            img_path = str(self.root_dir / line_parts[0])
            instances = [x.split(',') for x in line_parts[1:]]

            boxes = [tuple(map(int, map(float, x[:4]))) for x in instances]
            boxes = torch.tensor(boxes, dtype=torch.float32)

            labels = [int(x[4]) for x in instances]
            labels = torch.tensor(labels, dtype=torch.int64)

            target = {'boxes': boxes, 'labels': labels}
            self.img_paths.append(img_path)
            self.targets.append(target)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = F.to_tensor(Image.open(img_path).convert("L"))

        target = self.targets[idx]
        return image, target

    def __len__(self):
        return len(self.img_paths)
