
import os.path as osp

import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import GATConv, GCNConv
# import pytorch_lightning as pl

from os import path
from typing import Optional

import lightning.pytorch as pl
from lightning.pytorch import LightningDataModule, LightningModule, cli_lightning_logo
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.demos.mnist_datamodule import MNIST
from lightning.pytorch.utilities.imports import _TORCHVISION_AVAILABLE

from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, classification_report
import csv
import os
import time
from datetime import datetime
import json


class BaseModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.test_step_outputs = []
        self.log_version = ''
        self.test_start_time = None
        print("Base model initialised")

    def training_step(self, batch, batch_idx):
        y_hat = self(batch)
        if y_hat.shape[0] !=  batch.y.shape[0]:
            y_hat = self(batch, True)
            print(batch)
            print("y_hat", y_hat.shape)
            print("batch.y", batch.y.shape)
            exit()
        loss = F.cross_entropy(y_hat, batch.y)
        self.log("train_loss", loss, on_epoch=True, batch_size=batch.num_graphs)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self(batch)
        # print(batch)
        # print(y_hat.shape)
        # exit()
        loss = F.cross_entropy(y_hat, batch.y)
        self.log("valid_loss", loss, on_step=True, batch_size=batch.num_graphs)

    def test_step(self, batch, batch_idx):
        logits = self(batch)
        loss = F.cross_entropy(logits, batch.y)

        # # 对模型输出进行 softmax 操作，将 logits 转换为概率分布
        # probs = torch.softmax(y_hat, dim=1)
        # # 获取每个样本的预测类别
        # preds = torch.argmax(probs, dim=1)
        self.log("test_loss", loss, batch_size=batch.num_graphs)
        self.test_step_outputs.append({'logits': logits, 'labels': batch.y})
        return {'logits': logits, 'labels': batch.y}

    def on_test_epoch_start(self) -> None:
        print("=== on_test_epoch_start")
        self.test_start_time = time.time()

    def on_test_epoch_end(self):
        end_time = time.time()
        test_time = end_time - self.test_start_time
        print("Testing time (total)     :", test_time)
        print("Testing time (per sample):", test_time / len(self.test_step_outputs))
        formatted_time = str(datetime.utcfromtimestamp(test_time).strftime('%H:%M:%S.%f')[:-3])
        print("Testing time:", formatted_time)

        # 在测试周期结束时汇总所有测试步骤的结果并计算 F1 分数
        logits = torch.cat([output['logits'] for output in self.test_step_outputs], dim=0)
        labels = torch.cat([output['labels'] for output in self.test_step_outputs], dim=0)

        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        y_pred = preds.cpu()
        y_true = labels.cpu()

        # Predictions
        try:
            to_file = "lightning_logs/" + self.log_version + "/predictions.json"
            with open(to_file, 'w') as f:
                f.write('{{\n    "y_pred": {},\n    "y_true": {}\n}}'.format(
                    json.dumps(y_pred.numpy().tolist()),
                    json.dumps(y_true.numpy().tolist()),
                ))
            print("saved to", to_file)
        except Exception as e:
            print(str(e))

        # Classification Report
        try:
            report_txt = classification_report(y_true, y_pred, digits=4, output_dict=False, zero_division=1)
            print(report_txt)
            to_file = "lightning_logs/" + self.log_version + "/classification_report.txt"
            with open(to_file, "w") as f:
                f.write(report_txt)
            print("saved to", to_file)
        except Exception as e:
            print(str(e))

        try:
            report_dict = classification_report(y_true, y_pred, digits=4, output_dict=True, zero_division=1)
            to_file = "lightning_logs/" + self.log_version + "/classification_report.json"
            with open(to_file, "w") as f:
                f.write(json.dumps(report_dict))
            print("saved to", to_file)
            self.my_save_result_csv(report_dict)
        except Exception as e:
            print(str(e))

        f1 = f1_score(labels.cpu(), preds.cpu(), average='macro')
        self.test_step_outputs.clear()
        self.log('test_f1', f1)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        # x, _ = batch
        return self(batch)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def my_save_result_csv(self, report_dict, mode='a'):
        to_file_csv = self.res_file
        first = True
        if os.path.exists(to_file_csv):
            first = False
        float_num = 4
        with open(to_file_csv, mode, newline='\n') as file:
            writer = csv.writer(file)
            if first:
                writer.writerow(
                    ['Fold_id', 'F1-Ma', 'F1-Mi', 'F1-T', 'Pre-T', 'Rec-T', 'F1-F', 'Pre-F', 'Rec-F'])
            writer.writerow([
                self.fold_id,
                round(report_dict['macro avg']['f1-score'], float_num),
                round(report_dict['weighted avg']['f1-score'], float_num),
                round(report_dict['0']['f1-score'], float_num),
                round(report_dict['0']['precision'], float_num),

                round(report_dict['0']['recall'], float_num),
                round(report_dict['1']['f1-score'], float_num),
                round(report_dict['1']['precision'], float_num),
                round(report_dict['1']['recall'], float_num),
            ])

    def set_log_version(self, log_version):
        self.log_version = log_version


class GATNet(BaseModel):
    def __init__(self,
                 learning_rate=0.001,
                 num_layers=2,
                 hidden_dim=60,
                 weight_decay=5e-4,
                 num_classes=2,
                 dropout=0.6,
                 fold_id=0,
                 res_file='data/res.csv',
                 notes='',
                 ):
        super(GATNet, self).__init__()

        self.num_layers = num_layers
        self.hidden = hidden_dim
        self.learning_rate = learning_rate
        self.weight_decay =  weight_decay
        self.node_features = 384
        self.num_classes = num_classes
        self.dropout = dropout
        self.fold_id = fold_id
        self.res_file = res_file
        self.notes = notes

        self.conv1 = GCNConv(self.node_features, self.hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.convs.append(GCNConv(self.hidden, self.hidden))
        self.lin = torch.nn.Linear(self.hidden, self.num_classes)

        print("GATNet model initialised")


    def forward(self, data, debug=False):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.elu(conv(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = pyg_nn.global_mean_pool(x, data.batch)  # 全局池化操作
        x = self.lin(x)
        return F.log_softmax(x, dim=1)

    def my_save_result_csv(self, report_dict, mode='a'):
        to_file_csv = self.res_file
        first = True
        if os.path.exists(to_file_csv):
            first = False
        float_num = 4
        with open(to_file_csv, mode, newline='\n') as file:
            writer = csv.writer(file)
            if first:
                writer.writerow(
                    ['Fold_id', 'F1-Ma', 'F1-Mi', 'F1-T', 'Pre-T', 'Rec-T', 'F1-F', 'Pre-F', 'Rec-F',
                     'Log_version',
                     'LRate', 'Hidden', 'Layers', 'Dropout', 'Notes'])
            writer.writerow([
                self.fold_id,
                round(report_dict['macro avg']['f1-score'], float_num),
                round(report_dict['weighted avg']['f1-score'], float_num),
                round(report_dict['0']['f1-score'], float_num),
                round(report_dict['0']['precision'], float_num),

                round(report_dict['0']['recall'], float_num),
                round(report_dict['1']['f1-score'], float_num),
                round(report_dict['1']['precision'], float_num),
                round(report_dict['1']['recall'], float_num),

                self.log_version,
                self.learning_rate,
                self.hidden,
                self.num_layers,
                self.dropout,
                self.notes
            ])