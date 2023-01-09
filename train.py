import pytorch_lightning as pl

from configs import conf as configs
from dataset.dataloader import get_data_loader

from models.squeezenet import SqueezeFeatNet
from models.det_model import BarcodeDet
from models.lit_model import LitBarcodeDet
from dataset.transform import GeneralizedRCNNTransform


def main(conf):
    backbone = SqueezeFeatNet(in_channels=1)
    transform = GeneralizedRCNNTransform(min_size=800, max_size=1280, image_mean=0.5, image_std=0.22)
    det_model = BarcodeDet(backbone, 3, transform)
    lit_model = LitBarcodeDet(det_model)

    trainer = pl.Trainer(**vars(conf.train),
                         limit_train_batches=0.1,
                         limit_val_batches=0.1)
    train_loader, val_loader = get_data_loader(conf.data)

    trainer.fit(lit_model, train_loader, val_loader)
    trainer.test(lit_model, val_loader)


if __name__ == '__main__':
    main(configs)
