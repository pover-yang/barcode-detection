from torch.utils.data import DataLoader
from .dataset import BarcodeDataset, collate_fn


def get_data_loader(conf):
    train_dataset = BarcodeDataset(conf.data_root, mode='train')
    val_dataset = BarcodeDataset(conf.data_root, mode='test')

    train_loader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True,
                              num_workers=conf.num_workers, collate_fn=collate_fn,
                              pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=conf.batch_size, shuffle=False,
                            num_workers=conf.num_workers, collate_fn=collate_fn,
                            pin_memory=True, persistent_workers=True)
    return train_loader, val_loader
