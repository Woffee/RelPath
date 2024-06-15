import os
import json
from argparse import ArgumentParser, Namespace
import argparse
import torch
import numpy as np
from torch_geometric.data import Dataset, Data
import torch_geometric.loader as geom_loader
import random

class MyDataset(Dataset):
    def __init__(self, data_dir, transform=None, resampling=False, data_mode='all'):
        super(MyDataset, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "")
        self.data_dir = data_dir
        self.transform = transform
        self.data_list = []
        self.file_list = []
        self.mode = data_mode
        if self.mode == 'all':
            data_list = []
            for f in os.listdir(data_dir):
                if f.endswith(".pt"):
                    file_path = os.path.join(self.data_dir, f)
                    data_list.append(self.load_data(file_path))
            if resampling:
                print("resampling data")
                self.data_list = self.resample_data(data_list)
            else:
                self.data_list = data_list
        elif self.mode in ['train', 'valid', 'test']:
            data_list = []
            for f in os.listdir(data_dir):
                if f.endswith(".pt") and f.startswith(self.mode):
                    file_path = os.path.join(self.data_dir, f)
                    data_list.append(self.load_data(file_path))
            if resampling:
                print("resampling {} data".format(self.mode))
                self.data_list = self.resample_data(data_list)
            else:
                self.data_list = data_list
        else:
            for f in os.listdir(data_dir):
                if f.endswith(".pt"):
                    self.file_list.append(f)


    def __len__(self):
        if self.mode in ['all', 'train', 'valid', 'test']:
            return len(self.data_list)
        else:
            return len(self.file_list)

    def len(self):
        return self.__len__()

    def __getitem__(self, idx):
        if self.mode in ['all', 'train', 'valid', 'test']:
            return self.data_list[idx]
        else:
            file_name = self.file_list[idx]
            file_path = os.path.join(self.data_dir, file_name)
            return self.load_data(file_path)

    def load_data(self, file_path):
        d = torch.load(file_path)
        x = d['x'].to(self.device)
        edge_index = d['edge_index'].to(self.device)
        y = d['y']
        return Data(x=x, edge_index=edge_index, y=y)

    def get(self, idx):
        return self.__getitem__(idx)

    def resample_data(self, data_list):
        # 统计每个类别的样本数量
        class_counts = {}
        for data in data_list:
            label = data.y
            if label in class_counts:
                class_counts[label] += 1
            else:
                class_counts[label] = 1

        # 找到样本数量最多的类别
        max_count = max(class_counts.values())

        # 对样本数量较少的类别进行重复采样，直到与样本数量最多的类别一致
        resampled_data_list = []
        for label, count in class_counts.items():
            class_data = [data for data in data_list if data.y == label]
            if count < max_count:
                t = int(max_count / count)
                for i in range(t):
                    resampled_data_list.extend(class_data)

                repeat_data = random.choices(class_data, k=max_count % count)
                resampled_data_list.extend(repeat_data)
            else:
                resampled_data_list.extend(class_data)
        print("class_counts: ", class_counts)
        print("resampled_data_list: ", len(resampled_data_list))
        return resampled_data_list

if __name__ == '__main__':
    dataset = MyDataset('scripts/data/snopes_graph_v2')
    for i, d in enumerate(dataset):
        print(i, d.x.shape)