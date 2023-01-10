import pytorch_lightning as pl

from configs import conf as configs
from dataset.dataset_centernet import get_data_loader

from models.centernet import CenterNet


def main(conf):
    model = CenterNet()

    trainer = pl.Trainer(**vars(conf.train),
                         # limit_train_batches=1.0,
                         # limit_val_batches=0.1
                         )
    train_loader, val_loader = get_data_loader(conf.data)

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, val_loader)


if __name__ == '__main__':
    main(configs)
