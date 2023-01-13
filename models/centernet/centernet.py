import torch.nn as nn

from models.resnet import resnet50_feat, resnet18_feat


class CenterNet(nn.Module):
    def __init__(self, num_classes, img_channels, backbone='resnet18', pretrained_path=None, **kwargs):
        super(CenterNet, self).__init__()

        if backbone == 'resnet18':
            # 1, 1024, 1024 -> 512, 32, 32
            self.backbone = resnet18_feat(num_classes=10, img_channels=img_channels, pretrained_path=pretrained_path)
        elif backbone == 'resnet50':
            # 1, 1024, 1024 -> 2048, 32, 32
            self.backbone = resnet50_feat(num_classes=10, img_channels=img_channels)

        self.decoder = CenterNetDecoder(in_planes=512)  # 32, 32, 512 -> 256, 256, 8
        self.head = CenterNetHead(in_planes=8, num_classes=num_classes)  # 256, 256, 8 -> 256, 256, [num_classes|2|2]

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, images):
        feats = self.backbone(images)
        feats = self.decoder(feats)
        hmaps, whs, offsets = self.head(feats)
        return hmaps, whs, offsets


class CenterNetDecoder(nn.Module):
    def __init__(self, in_planes):
        super(CenterNetDecoder, self).__init__()
        self.in_planes = in_planes

        # ----------------------------------------------------------#
        #   32,32,512 -> 64,64,128 -> 128,128,32 -> 256,256,8
        #   利用ConvTranspose2d进行上采样。
        #   每次特征层的宽高变为原来的两倍。
        # ----------------------------------------------------------#
        self.deconv_layers = self.make_deconv_layer(
            num_layers=3,
            out_planes_list=[128, 32, 8],
            kernels_size_list=[4, 4, 4],
        )

    def make_deconv_layer(self, num_layers, out_planes_list, kernels_size_list):
        layers = []
        for i in range(num_layers):
            kernel_size = kernels_size_list[i]
            out_planes = out_planes_list[i]

            layers.append(
                nn.ConvTranspose2d(in_channels=self.in_planes, out_channels=out_planes,
                                   kernel_size=kernel_size, stride=2, padding=1,
                                   output_padding=0, bias=False))
            layers.append(nn.BatchNorm2d(out_planes))
            layers.append(nn.ReLU(inplace=True))

            self.in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.deconv_layers(x)


class CenterNetHead(nn.Module):
    def __init__(self, in_planes, num_classes):
        super(CenterNetHead, self).__init__()
        # 热力图预测部分
        self.hmap_head = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes, num_classes, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        # 宽高预测的部分
        self.wh_head = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes, 2, kernel_size=1, stride=1, padding=0)
        )

        # 中心点预测的部分
        self.offset_head = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes, 2, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        hmap = self.hmap_head(x)
        wh = self.wh_head(x)
        offset = self.offset_head(x)
        return hmap, wh, offset


if __name__ == '__main__':
    model = CenterNet(num_classes=10, img_channels=1, backbone='resnet18', pretrained_path=None)
