from argparse import Namespace
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
from utils import next_version

# ---------- Configs ----------
log_root = '/home/junjieyang/ExpLogs/'
exp_name = 'CenterNet'
version = next_version(log_root+exp_name)


conf = Namespace(data=Namespace(), model=Namespace(), train=Namespace())

# data conf
conf.data.batch_size = 10
conf.data.num_workers = 16
conf.data.root_dir = "/home/junjieyang/Data/Barcode-Detection-Data"

# models conf
conf.model.head_lr = 1e-3
conf.model.backbone_lr = 1e-5
conf.model.num_classes = 3
conf.model.img_channels = 1
conf.model.backbone = 'resnet18'
# conf.model.pretrained_path = '/home/junjieyang/PyProjs/barcode-detection/ckpt/resnet18-epoch=42-val_loss=0.54.pth'


# train conf
conf.train.devices = [0, 1, 2, 3, 4, 5, 6, 7]
conf.train.accelerator = 'gpu'
conf.train.max_epochs = 100
conf.train.profiler = 'simple'
conf.train.default_root_dir = log_root
conf.train.limit_train_batches = 1.
conf.train.limit_val_batches = 1.
conf.train.strategy = DDPStrategy(find_unused_parameters=False)
conf.train.logger = TensorBoardLogger(log_root, name=exp_name, version=f'v{version}')
# ---------- Configs ----------
