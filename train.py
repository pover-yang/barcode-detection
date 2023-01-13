import pytorch_lightning as pl

from configs.centernet_cf import conf
from dataset.centernet_ds import centernet_dataloader

from models import LitCenterNet


def main():
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    ckpt_path = '/home/junjieyang/PyProjs/barcode-detection/' \
                'ckpt/centernet-resnet18-epoch=55-val_loss=4.38.ckpt'
    model = LitCenterNet(conf.model).load_from_checkpoint(ckpt_path, model_conf=conf.model)

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    trainer = pl.Trainer(**vars(conf.train))

    # ------------------------
    # 3 START TRAINING
    # ------------------------

    train_loader, val_loader = centernet_dataloader(conf.data)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, val_loader)


if __name__ == '__main__':
    main()
