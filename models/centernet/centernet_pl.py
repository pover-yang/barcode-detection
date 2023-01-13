import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, BackboneFinetuning

from models import CenterNet
from utils import warmup_lr, centernet_loss


class LitCenterNet(CenterNet, pl.LightningModule):
    def __init__(self, model_conf):
        super().__init__(**vars(model_conf))
        self.init_weights()
        self.conf = model_conf

    def training_step(self, batch, batch_idx):
        images, targets = batch
        hmap, wh, offset = self.forward(images)
        loss_dict = centernet_loss(hmap, wh, offset, targets)
        loss = loss_dict['hmap_loss'] + 0.1 * loss_dict['wh_loss'] + loss_dict['off_loss']
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        hmap, wh, offset = self.forward(images)
        loss_dict = centernet_loss(hmap, wh, offset, targets)
        loss = loss_dict['hmap_loss'] + 0.1 * loss_dict['wh_loss'] + loss_dict['off_loss']
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, targets = batch
        hmap, wh, offset = self.forward(images)
        loss_dict = centernet_loss(hmap, wh, offset, targets)
        loss = loss_dict['hmap_loss'] + 0.1 * loss_dict['wh_loss'] + loss_dict['off_loss']
        return loss

    def configure_optimizers(self):
        # params excluded backbone
        # params = [params for name, params in self.named_parameters() if not name.startswith('backbone')]
        params = self.parameters()

        lr_lambda = warmup_lr(max_epochs=self.trainer.max_epochs, warmup_epochs=10)
        optimizer = torch.optim.Adam(params, lr=self.conf.head_lr)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return [optimizer], [lr_scheduler]

    def configure_callbacks(self):
        lr_monitor = LearningRateMonitor(logging_interval='epoch')

        backbone_finetune = BackboneFinetuning(
            unfreeze_backbone_at_epoch=20,
            lambda_func=warmup_lr(max_epochs=self.trainer.max_epochs, warmup_epochs=40),
            backbone_initial_lr=self.conf.backbone_lr,
            should_align=False,
        )
        model_checkpoint = ModelCheckpoint(
            monitor='val_loss',
            filename='centernet-resnet18-{epoch:02d}-{val_loss:.2f}',
            save_top_k=200,
            mode='min',
        )

        # return [lr_monitor, backbone_finetune, model_checkpoint]
        return [lr_monitor, model_checkpoint]


def cal_flops():
    import torch
    from thop import profile
    from configs.centernet_cf import conf

    # Evaluate the model's flops
    model = LitCenterNet(conf.model)
    input_tensor = torch.randn(1, 3, 384, 640)
    flops, params = profile(model, inputs=(input_tensor,))
    print(f"Flops: {flops / 1e9:.3f}G, Params: {params / 1e6:.3f}M")


def model_infer():
    import time
    from torch.utils.data import DataLoader
    from dataset.centernet_ds import CenterNetDataset, collate_fn
    from configs.centernet_cf import conf

    # Evaluate the time of model forward
    model = LitCenterNet(**vars(conf.model)).load_from_checkpoint('../ckpt/epoch=199-step=75000.ckpt')

    dataset = CenterNetDataset(r'D:\Barcode-Detection-Data', mode='test')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=collate_fn)
    for i, (images, targets) in enumerate(dataloader):
        start = time.time()
        hmap, wh, offset = model(images)
        loss_dict = model.loss(hmap, wh, offset, targets)
        print(f"Forward time: {time.time() - start:.3f}s")
        print(f"Loss: {loss_dict}")
        break


if __name__ == "__main__":
    cal_flops()
    # model_infer()
