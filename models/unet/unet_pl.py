import pytorch_lightning as pl
from models.unet import UNet
from utils import focal_loss


class LitUnet(UNet, pl.LightningModule):
    def __init__(self, model_conf):
        super().__init__(**vars(model_conf))

    def loss(self, outputs, targets):
        return self.criterion(outputs, targets)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.forward(images)
        loss = self.loss(outputs, targets)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss
