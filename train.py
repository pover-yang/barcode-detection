import os
import pytorch_lightning as pl

from dataset.heatmap_ds import train_val_loader

from models import LitUNet

if os.name == 'nt':
    from configs.unet_cf_win import conf
else:
    from configs.unet_cf_linux import conf


def main():
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = LitUNet(conf.model)

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    trainer = pl.Trainer(**vars(conf.train))

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    train_loader, val_loader = train_val_loader(conf.data)
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    main()
