import torch
import math

import pytorch_lightning as pl
from torchvision.models.detection import fasterrcnn_resnet50_fpn


def warmup_lr(max_epochs, warmup_epochs=None, warmup_factor=0.1):
    if not warmup_epochs:
        warmup_epochs = max_epochs // 20

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return warmup_factor + (1 - warmup_factor) * epoch / warmup_epochs
        else:
            return 1 / 2 * (1 + math.cos((epoch - warmup_epochs) / (max_epochs - warmup_epochs) * math.pi))

    return lr_lambda


class LitBarcodeDet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = fasterrcnn_resnet50_fpn(num_classes=3)

    def training_step(self, batch):
        images, targets = batch
        loss_dict, detections = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        return losses

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict, detections = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        self.log('val_loss', float(losses), sync_dist=True)
        return losses

    def test_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict, detections = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        return losses

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-6)
        lr_lambda = warmup_lr(max_epochs=self.trainer.max_epochs)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def configure_callbacks(self):
        from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
        ckpt_callback = ModelCheckpoint(filename='barcode-det-{epoch:03d}-{loss:.7f}',
                                        monitor='val_loss', save_top_k=300, mode='min')
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        return [lr_monitor, ckpt_callback]


if __name__ == '__main__':
    from det_model import BarcodeDet
    from squeezenet import SqueezeFeatNet
    from dataset.transform import GeneralizedRCNNTransform
    backbone = SqueezeFeatNet(in_channels=1)
    transform = GeneralizedRCNNTransform(min_size=800, max_size=1280, image_mean=0.5, image_std=0.22)
    det_model = BarcodeDet(backbone, 3, transform)
    lit_model = LitBarcodeDet(det_model)
    pass