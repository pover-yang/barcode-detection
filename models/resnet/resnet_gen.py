import torch
import torch.nn as nn

from models.resnet import ResNet, BasicBlock, Bottleneck


def resnet50_feat(num_classes, img_channels):
    # 1, 1024, 1024 -> 2048, 32, 32
    # Flops: 84.698G, Params: 23.502M
    # model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes, img_channels=img_channels)
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes, img_channels=img_channels)  # old,
    model = nn.Sequential(*list(model.children())[:-2])
    return model


def resnet18_feat(num_classes, img_channels, pretrained_path=None):
    # 1, 1024, 1024 -> 512, 32, 32
    # Flops: 36.463G, Params: 11.170M
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, img_channels=img_channels)
    if pretrained_path:
        model.load_state_dict(torch.load(pretrained_path), strict=True)
    model = nn.Sequential(*list(model.children())[:-2])
    return model


def resnet18(num_classes, img_channels, pretrained_path=None):
    # 1, 1024, 1024 -> 512, 32, 32
    # Flops: 36.463G, Params: 11.170M
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, img_channels=img_channels)
    if pretrained_path:
        model.load_state_dict(torch.load(pretrained_path), strict=True)
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
