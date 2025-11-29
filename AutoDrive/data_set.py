import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class DataSet(Dataset):
    def __init__(self, data_folder, mode, transform=None):
        self.data_folder = data_folder
        self.mode = mode.lower()
        self.transform = transform

        assert self.mode in {'train', 'evaluation'}

        if self.mode == 'train':
            file_path = os.path.join(self.data_folder, 'train.txt')
        else:
            file_path = os.path.join(self.data_folder, 'evaluation.txt')

        self.file_list = list()
        with open(file_path, 'r') as f:
            files = f.readline()

            self.file_list.append([files.split(" ", 1)[0], float(files.split(" ", 1)[1])])

    def __getitem__(self, item):
        img = cv2.imread(self.file_list[item][0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        if self.transform:
            img = self.transform(img)

        label = self.file_list[item][1]
        label = torch.from_numpy(np.array([label])).float()
        return img, label

    def __len__(self):
        return len(self.file_list)