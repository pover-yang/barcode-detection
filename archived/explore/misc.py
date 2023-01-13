import cv2
import torch
import numpy as np
import torch.nn as nn
import torchpwl
import torchvision.transforms.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN
from torchvision.models import resnet50
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt


class GammaCorrection(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.tensor(1.499527931213379))

    def forward(self, x):
        x = x.sign() * (x.abs() + self.eps) ** self.gamma
        return [x]



class PWLToneMapping(nn.Module):
    def __init__(self, num_breakpoints=100):
        super().__init__()
        self.pwl = torchpwl.PWL(num_channels=1, num_breakpoints=num_breakpoints)
        # self.pwl = torchpwl.PWL(num_channels=1, num_breakpoints=num_breakpoints)

    def forward(self, x):
        x_flatten = x.flatten(start_dim=1).transpose(0, 1)
        y_flatten = self.pwl(x_flatten).clip(0, 1)
        out = y_flatten.reshape(x.shape)
        return [out]


def plot(pwl, x_start=-10, x_end=10):
    fig = plt.figure(figsize=(8, 8))
    plt.ylim((x_start, x_end))
    plt.xlim((x_start, x_end))
    x = torch.linspace(x_start, x_end, steps=1000).unsqueeze(1)
    y = pwl(x)
    plt.plot(list(x), list(y), "b")

    x = x.detach()
    y = y.detach()
    y = y.detach()
    plt.plot(list(x), list(y), "b")

    plt.plot(list(pwl.x_positions.squeeze(0).detach()), list(pwl(pwl.x_positions.view(-1, 1)).squeeze(1).detach()),
             "or", )
    fig.canvas.draw()
    curve = np.array(fig.canvas.renderer.buffer_rgba())
    return curve
