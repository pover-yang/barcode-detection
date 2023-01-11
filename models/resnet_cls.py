import pytorch_lightning as pl
from resnet import resnet18
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from torchvision import transforms


class ResNet18(pl.LightningModule):
    def __init__(self, num_classes, img_channels=1):
        super(ResNet18, self).__init__()
        self.model = resnet18(num_classes=num_classes, img_channels=img_channels)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}


def ckpt_to_pth(ckpt_path):
    pl_model = ResNet18(num_classes=10, img_channels=1)
    model = pl_model.model
    torch.save(pl_model.model.state_dict(), ckpt_path.replace('.ckpt', '.pth'))


def train():
    model = ResNet18(num_classes=10, img_channels=1)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Grayscale(num_output_channels=1),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Grayscale(num_output_channels=1),
    ])

    train_dataset = CIFAR10(root='/home/junjieyang/Data/CIFAR10', train=True, download=True, transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True, num_workers=16, persistent_workers=True)

    test_dataset = CIFAR10(root=r'/home/junjieyang/Data/CIFAR10', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False, num_workers=16, persistent_workers=True)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath='/home/junjieyang/ExpLogs/ImgCls',
        filename='resnet18-{epoch:02d}-{val_loss:.2f}',
        save_top_k=100,
        mode='min',
    )

    trainer = pl.Trainer(
        devices=[1, 2, 3, 4, 5, 6, 7],
        accelerator='gpu',
        max_epochs=500,
        default_root_dir='/home/junjieyang/ExpLogs/ImgCls',
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model, train_loader, test_loader)

    print('done')


if __name__ == '__main__':
   # train()
    ckpt_to_pth('../ckpt/resnet18-epoch=42-val_loss=0.54.ckpt')