import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.ops
from torch.utils.data import DataLoader

from dataset_centernet import CenterNetDataset, collate_fn
from dataset_torchvision import TorchvisionDataset


def draw_img_with_labels(image, target):
    label_name = {
        0: '1d',
        1: 'qr',
        2: 'dm'
    }
    if type(target) == dict:
        label_strs = [label_name[int(x)] for x in target['labels']]
        boxes = target['boxes']
    else:
        label_strs = [label_name[int(x[4])] for x in target]
        boxes = [list(map(int, x[:4])) for x in target]
        boxes = torch.tensor(boxes, dtype=torch.float32)

    image = torch.asarray(image * 255, dtype=torch.uint8)
    image = torchvision.utils.draw_bounding_boxes(image, boxes, labels=label_strs, colors=(255, 0, 0), width=2)
    image = np.asarray(image).transpose((1, 2, 0))
    return image


def draw_img_with_heatmap(image, heatmap):
    image = np.asarray(image * 255, dtype=np.uint8).transpose((1, 2, 0))
    image = np.concatenate([image, image, image], axis=2)

    heatmap = np.asarray(heatmap * 255, dtype=np.uint8).transpose((1, 2, 0))
    heatmap = cv2.resize(heatmap, (image.shape[0], image.shape[1]))

    blended_image = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)
    return blended_image


def vis_bbox():
    dataset = TorchvisionDataset(r'D:\Barcode-Detection-Data', r'D:\Barcode-Detection-Data\all.txt')

    # 从dataset中随机取出20张图片，画出对应的bbox和label
    indices = np.random.randint(0, len(dataset), 20)
    vis_imgs = [draw_img_with_labels(*dataset[i]) for i in indices]
    # 画出20张图片
    fig, axes = plt.subplots(4, 5, figsize=(20, 15))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(vis_imgs[i])
        ax.axis('off')
    plt.show()


def vis_hmap(root_dir):
    dataset = CenterNetDataset(root_dir, mode='all', input_size=1024, num_classes=3, filter_labels=[0])

    indices = np.random.randint(0, len(dataset), 20)
    fig, axes = plt.subplots(4, 5, figsize=(10, 8))

    vis_imgs = []
    for i in indices:
        image, target = dataset[i]
        # 画出heatmap
        vis_img = draw_img_with_heatmap(image, target['hmap'])
        vis_imgs.append(vis_img)

    # 画出20张图片
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(vis_imgs[i])
        ax.axis('off')
    plt.show()


def dataloader_test():
    dataset = CenterNetDataset(r'D:\Barcode-Detection-Data', mode='all')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)
    for i, (images, targets) in enumerate(dataloader):
        print(images.shape)
        print(targets['hmap'].shape)
        # print(targets['wh'].shape)
        # print(targets['offset'].shape)
        # break
        pass


def cal_mean_std(root_dir):
    dataset = CenterNetDataset(root_dir, mode='all')
    dataloader = DataLoader(dataset, batch_size=30, shuffle=False, num_workers=15, collate_fn=collate_fn)

    mean = torch.zeros(1)
    var = torch.zeros(1)
    num_samples = 0.
    for images, _ in dataloader:
        batch_samples = images.size(0)
        # Rearrange batch to be the shape of [B, C, W * H]
        images = images.view(batch_samples, images.size(1), -1)

        # Update total number of images
        num_samples += batch_samples

        # Compute mean and std here
        mean += images.mean(2).sum(0)
        var += images.var(2).sum(0)

        if num_samples % 1000 == 0:
            print(num_samples, mean / num_samples, torch.sqrt(var / num_samples))

    mean /= num_samples
    var /= num_samples
    std = torch.sqrt(var)

    print(f'mean: {mean}, std: {std}')
    with open(f'{root_dir}/mean_std.txt', 'w', encoding='utf-8') as f:
        f.write(f'mean: {mean}, std: {std}')


if __name__ == '__main__':
    root = r'D:\Barcode-Detection-Data'

    # vis_bbox()
    vis_hmap(root)
    # dataloader_test()
    # cal_mean_std(root)
