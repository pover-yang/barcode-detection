import torch
import torch.nn as nn
import pytorch_lightning as pl

from models.resnet import resnet50_feat
from utils.centernet_loss import focal_loss, reg_l1_loss


class CenterNet(pl.LightningModule):
    def __init__(self, num_classes=3, img_channel=1):
        super(CenterNet, self).__init__()

        self.backbone = resnet50_feat(num_classes, img_channel)  # 1024, 1024, 1 -> 32, 32, 512
        self.decoder = Decoder(in_planes=512)  # 32, 32, 512 -> 256, 256, 8
        self.head = Head(num_classes, in_planes=8)  # 256, 256, 8 -> 256, 256, [num_classes|2|2]

        self.init_weights()

        self.focal_loss = focal_loss
        self.reg_l1_loss = reg_l1_loss

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

    def loss(self, hmap, wh, offset, target):
        hmap_loss = self.focal_loss(hmap, target['hmap'])
        wh_loss = self.reg_l1_loss(wh, target['wh'], target['reg_mask'])
        offset_loss = self.reg_l1_loss(offset, target['offset'], target['reg_mask'])

        loss_dict = {'hmap_loss': hmap_loss, 'wh_loss': wh_loss, 'off_loss': offset_loss}
        return loss_dict

    def training_step(self, batch, batch_idx):
        images, targets = batch
        hmap, wh, offset = self(images)
        loss_dict = self.loss(hmap, wh, offset, targets)
        loss = sum(loss_dict.values())
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        hmap, wh, offset = self(images)
        loss_dict = self.loss(hmap, wh, offset, targets)
        loss = sum(loss_dict.values())
        return loss

    def test_step(self, batch, batch_idx):
        images, targets = batch
        hmap, wh, offset = self(images)
        loss_dict = self.loss(hmap, wh, offset, targets)
        loss = sum(loss_dict.values())
        return loss

    def configure_optimizers(self):
        import math
        def warmup_lr(max_epochs, warmup_epochs=None, warmup_factor=0.1):
            if not warmup_epochs:
                warmup_epochs = max_epochs // 20

            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return warmup_factor + (1 - warmup_factor) * epoch / warmup_epochs
                else:
                    return 1 / 2 * (1 + math.cos((epoch - warmup_epochs) / (max_epochs - warmup_epochs) * math.pi))

            return lr_lambda

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-6)
        lr_lambda = warmup_lr(max_epochs=self.trainer.max_epochs)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}


class Decoder(nn.Module):
    def __init__(self, in_planes):
        super(Decoder, self).__init__()
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


class Head(nn.Module):
    def __init__(self, num_classes, in_planes):
        super(Head, self).__init__()
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


def main():
    import torch
    import time
    from thop import profile
    from torch.utils.data import DataLoader
    from dataset.dataset import BarcodeCenterNetDataset, collate_fn

    # Evaluate the model's flops
    model = CenterNet()
    input_tensor = torch.randn(1, 1, 1024, 1024)
    flops, params = profile(model, inputs=(input_tensor,))
    print(f"Flops: {flops / 1e9:.3f}G, Params: {params / 1e6:.3f}M")

    # Evaluate the time of model forward
    dataset = BarcodeCenterNetDataset('/Users/yjunj/Data/Barcode-Detection-Data/data',
                                      '/Users/yjunj/Data/Barcode-Detection-Data/test.txt')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)
    for i, (images, targets) in enumerate(dataloader):
        start = time.time()
        hmap, wh, offset = model(images)
        loss_dict = model.loss(hmap, wh, offset, targets)
        print(f"Forward time: {time.time() - start:.3f}s")
        print(f"Loss: {loss_dict}")
        break


if __name__ == "__main__":
    main()
