from collections import OrderedDict

import torch
import torch.nn as nn


class Fire(nn.Module):
    def __init__(self, in_planes: int, squeeze_planes: int, expand1x1_planes: int, expand3x3_planes: int) -> None:
        super().__init__()
        self.squeeze = nn.Conv2d(in_planes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat(
            [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        )
    
    
class SqueezeFeatNet(nn.Module):
    def __init__(self,
                 in_channels: int = 3) -> None:
        super().__init__()
        self.out_channels = 512
        # _log_api_usage_once(self)

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        )

        self.block_1 = nn.Sequential(
            Fire(64, 16, 64, 64),
            Fire(128, 16, 64, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        )

        self.block_2 = nn.Sequential(
            Fire(128, 32, 128, 128),
            Fire(256, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        )

        self.block_3 = nn.Sequential(
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            Fire(512, 64, self.out_channels // 2, self.out_channels // 2),
        )

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),

            Fire(64, 16, 64, 64),
            Fire(128, 16, 64, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),

            Fire(128, 32, 128, 128),
            Fire(256, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),

            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            Fire(512, 64, self.out_channels//2, self.out_channels//2),
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        x1 = self.in_conv(x)
        x2 = self.block_1(x1)
        x3 = self.block_2(x2)
        x4 = self.block_3(x3)
        # x = self.features(x)
        return OrderedDict([
            ('1', x1),
            ('2', x2),
            ('3', x3),
            ('4', x4)
        ])
        # return x1, x2, x3, x4
