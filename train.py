
import os.path
from os import path
from typing import Optional

import torch
from lightning.pytorch import LightningDataModule, LightningModule, cli_lightning_logo
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.demos.mnist_datamodule import MNIST
from lightning.pytorch.utilities.imports import _TORCHVISION_AVAILABLE
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint

from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Data, Batch

from MyDataset import MyDataset
from model import GATNet
import argparse
from sklearn.metrics import classification_report
import pandas as pd
import matplotlib.pyplot as plt
import math
from datetime import datetime
import time
from utils import log_message

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms

def my_custom_collate(batch):
    # 检查 batch 中是否包含 torch_geometric.data.Data 对象
    if isinstance(batch[0], Data):
        # 将 Data 对象列表转换为 Batch 对象
        return Batch.from_data_list(batch)
    else:
        # 对于其他类型的数据，使用默认的 collate 函数进行处理
        return default_collate(batch)

class MyDataModule(LightningDataModule):
    def __init__(self,
                 batch_size: int = 32,
                 data_set_name: str = 'snopes_graph_v5',
                 resampling: bool = False,
                 data_mode: str = 'all'
                 ):
        super().__init__()

        print("=== data_set_name:", data_set_name)
        if data_set_name.endswith('split'):
            print("read split data")
            self.data_train = MyDataset(path.join("./scripts/data", data_set_name),
                                        resampling=True,
                                        data_mode='train')
            self.data_val = MyDataset(path.join("./scripts/data", data_set_name),
                                        resampling=True,
                                        data_mode='valid')
            self.data_test = MyDataset(path.join("./scripts/data", data_set_name),
                                        resampling=False,
                                        data_mode='test')
        elif data_set_name.find('politifact') > -1:
            print("read politifact data")
            dataset = MyDataset(path.join("./scripts/data", data_set_name), resampling=resampling, data_mode=data_mode)
            print("loading dataset from: {}".format(data_set_name))
            self.data_train, self.data_val, self.data_test = random_split(
                dataset, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42)
            )
        else:
            dataset = MyDataset(path.join("./scripts/data", data_set_name), resampling=resampling, data_mode=data_mode)
            print("loading dataset from: {}".format(data_set_name))

            self.data_train, self.data_val, _ = random_split(
                dataset, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42)
            )
            dataset_original = MyDataset(path.join("./scripts/data", data_set_name), resampling=False, data_mode=data_mode)
            _2, _3, self.data_test = random_split(
                dataset_original, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42)
            )
        print("data_train size: ", len(self.data_train))
        print("data_val size: ", len(self.data_val))
        print("data_test size: ", len(self.data_test))
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, collate_fn=my_custom_collate)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, collate_fn=my_custom_collate)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, collate_fn=my_custom_collate)

    def predict_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, collate_fn=my_custom_collate)

def draw_log(log_dir):
    # 加载 metrics.csv 文件
    metrics_df = pd.read_csv(os.path.join(log_dir, 'metrics.csv'))

    epochs = []
    train_loss_epoch = []
    valid_loss_epoch = []
    for i, row in metrics_df.iterrows():
        if not math.isnan(row['train_loss_epoch']):
            epochs.append(row['epoch'])
            train_loss_epoch.append(row['train_loss_epoch'])
        if not math.isnan(row['valid_loss_epoch']):
            valid_loss_epoch.append(row['valid_loss_epoch'])

    # 绘制训练损失和验证损失曲线
    plt.plot(epochs, train_loss_epoch, label='Train Loss (Epoch)')
    plt.plot(epochs, valid_loss_epoch, label='Valid Loss (Epoch)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join(log_dir, 'train_loss_epoch.pdf'))
    print("saved plot to {}".format(os.path.join(log_dir, 'train_loss_epoch.pdf')))

def cli_main():
    log_version = "version_" + datetime.now().strftime("%m%d-%H%M%S")
    ckpt_dir = '/project/wu/ww6/EvolveFC_CKPT/{}'.format(log_version)
    if not path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    cli = LightningCLI(
        GATNet,
        MyDataModule,
        seed_everything_default=1234,
        save_config_kwargs={"overwrite": True},
        trainer_defaults={
            'logger': {'class_path': 'lightning.pytorch.loggers.CSVLogger',
                       'init_args': {'save_dir': './', 'version': log_version}},
            # "max_epochs": 15,
            # "limit_train_batches": 15,
            # "limit_val_batches": 15,
            # "accelerator": "gpu",
            "callbacks": [
                EarlyStopping(monitor='valid_loss', patience=50, mode='min'),  # 添加 EarlyStopping 回调
                ModelCheckpoint(
                    monitor='valid_loss',
                    # filename='{epoch}-{valid_loss:.2f}',
                    save_top_k=3,
                    mode='min'
                )
            ]
        },
        run=False
    )
    cli.model.set_log_version(log_version)

    train_start_time = time.time()
    log_message("=== train start ===")
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    log_message("=== train end ===")

    train_time = time.time() - train_start_time
    print("Train time (total)     :", train_time)
    formatted_time = str(datetime.utcfromtimestamp(train_time).strftime('%H:%M:%S.%f')[:-3])
    print("Train time (total)     :", formatted_time)

    log_message("=== test start ===")
    cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)
    log_message("=== test end ===")

    draw_log("lightning_logs/"+log_version)
    print("res_file:", cli.model.res_file)
    print("ckpt_dir:", ckpt_dir)
    print("log_dir:", "lightning_logs/" + log_version)

    print("done")


def check_gpu():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU info:")
        print("GPU device name     :", torch.cuda.get_device_name(0))
        print("GPU device count    :", torch.cuda.device_count())
        print("GPU current_device  :", torch.cuda.current_device())
        print("GPU total memory(GB):", round(torch.cuda.get_device_properties(device).total_memory / (1024**3), 1))
    else:
        print("GPU not available")

if __name__ == "__main__":
    log_message("= main start")
    sh_start_time = time.time()

    # cli_lightning_logo()
    torch.set_float32_matmul_precision('medium')

    check_gpu()

    cli_main()

    sh_end_time = time.time()
    formatted_time = str(datetime.utcfromtimestamp(sh_end_time - sh_start_time).strftime('%H:%M:%S.%f')[:-3])

    log_message("= main end")
    print("Total sh time (total): ", formatted_time)