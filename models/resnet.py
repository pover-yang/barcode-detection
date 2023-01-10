import torch
import torch.nn as nn
from torch import Tensor


def conv1x1(in_planes, out_planes, stride=1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.bn1 = nn.BatchNorm2d(out_planes)

        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, out_planes, stride=1, downsample=None):
        super().__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(in_planes, out_planes)
        self.bn1 = nn.BatchNorm2d(out_planes)

        self.conv2 = conv3x3(out_planes, out_planes, stride)
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.conv3 = conv1x1(out_planes, out_planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(out_planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, img_channels=3, in_planes=64):
        super().__init__()

        # Input layer
        self.in_planes = in_planes
        self.conv1 = nn.Conv2d(img_channels, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # middle layers
        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)

        # Output layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, block, out_planes, num_blocks, stride=1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_planes != out_planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_planes, out_planes * block.expansion, stride),
                nn.BatchNorm2d(out_planes * block.expansion),
            )

        layers = [block(self.in_planes, out_planes, stride, downsample)]
        self.in_planes = out_planes * block.expansion

        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, out_planes, stride=1, downsample=None))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        # Input
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Middle layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Output
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet50_feat(num_classes, img_channels):
    # 1, 1024, 1024 -> 2048, 32, 32
    # Flops: 84.698G, Params: 23.502M
    # model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes, img_channels=img_channels)
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes, img_channels=img_channels)  # old,
    model = nn.Sequential(*list(model.children())[:-2])
    return model


def resnet18_feat(num_classes, img_channels):
    # 1, 1024, 1024 -> 512, 32, 32
    # Flops: 36.463G, Params: 11.170M
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, img_channels=img_channels)
    model = nn.Sequential(*list(model.children())[:-2])
    return model


def resnet18(num_classes, img_channels):
    # 1, 1024, 1024 -> 512, 32, 32
    # Flops: 36.463G, Params: 11.170M
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, img_channels=img_channels)
    return model

if __name__ == "__main__":
    import torch
    from thop import profile

    img_c = 1
    # Evaluate the model's flops
    # net = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=3, img_channels=img_c)  # resnet26
    # net = nn.Sequential(*list(net.children())[:-2])

    net = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=3, img_channels=img_c)
    # net = nn.Sequential(*list(net.children())[:-2])

    input_tensor = torch.randn(1, img_c, 1024, 1024)
    flops, params = profile(net, inputs=(input_tensor,))
    print(f"Flops: {flops / 1e9:.3f}G, Params: {params / 1e6:.3f}M")
