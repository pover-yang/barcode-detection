import torch.nn as nn
from resnet import resnet50_feat


class CenterNet(nn.Module):
    def __init__(self, num_classes=3, img_channel=1):
        super(CenterNet, self).__init__()

        # 1024, 1024, 1 -> 16,16,2048
        self.backbone = resnet50_feat(num_classes, img_channel)

        self.init_weights()

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        feat = self.backbone(x)
        decode = self.decoder(feat)
        out = self.head(decode)
        return out



