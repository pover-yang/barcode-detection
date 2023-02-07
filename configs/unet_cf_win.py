from argparse import Namespace
from pytorch_lightning.loggers import TensorBoardLogger
from utils import next_version
from dataset.heatmap_ds import sample_loader

# ---------- Configs ----------
log_root = r'D:/ExpLogs'
exp_name = 'UNet'
version = next_version(log_root, exp_name)


conf = Namespace(data=Namespace(), model=Namespace(), train=Namespace())

# data conf
conf.data.batch_size = 8
conf.data.num_workers = 16
conf.data.root_dir = r'D:/Barcode-Detection-Data'

# models conf
conf.model.in_channels = 1
conf.model.n_classes = 3
conf.model.inc_channels = 8
conf.model.sample_loader = sample_loader(conf.data)

# train conf
conf.train.accelerator = 'gpu'
conf.train.max_epochs = 100
conf.train.profiler = 'simple'
conf.train.default_root_dir = log_root
conf.train.limit_train_batches = 0.05
conf.train.limit_val_batches = 0.05
conf.train.logger = TensorBoardLogger(log_root, name=exp_name, version=f'v{version}')
# ---------- Configs ----------
