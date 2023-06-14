import os
import torch
from torch.utils.data import Dataset
import netCDF4
import numpy as np


class OceanDataset(Dataset):

    def __init__(self,
                 data_file,
                 tslag=3,
                 spinupts=0,
                 fine_tune=False,
                 device='cpu',
                 multi_expt_normalize=False,
                 mmap_mode=None):
        self.tslag = tslag
        self.spinupts = spinupts

        data_dir = os.path.dirname(data_file)

        if device == 'cpu':
            mmap_mode = 'r'

        self.data = np.load(data_file, mmap_mode=mmap_mode)[spinupts:]
        if multi_expt_normalize is True:
            stats_file = np.load(
                os.path.join(data_dir, "dynDiagsGlobalStats2D.npz"))
        else:
            stats_file = np.load(os.path.join(data_dir, "dynDiagsStats.npz"))
        self.means = stats_file['timemeans']
        self.stdevs = stats_file['timestdevs']

        self.img_size = [self.data.shape[-1], self.data.shape[-2]]
        self.channels = self.data.shape[1]
        self.fine_tune = fine_tune
        self.means = np.concatenate(
            (self.means, np.zeros(
                (1, self.data.shape[-1], self.data.shape[-2]))),
            axis=0)
        self.stdevs = np.concatenate(
            (self.stdevs, np.ones(
                (1, self.data.shape[-1], self.data.shape[-2]))),
            axis=0)
        if self.fine_tune:
            self.len_ = self.data.shape[0] - self.tslag * 2
            self.getitem = self.getitem_finetune
        else:
            self.len_ = self.data.shape[0] - self.tslag
            self.getitem = self.getitem_nofinetune

        # self.transform = transforms.Normalize(mean=self.means, std=self.stdevs)
        # self.target_transform = transforms.Normalize(mean=self.means,
        self.transform = lambda x: (x - self.means) / (self.stdevs + 1e-5)
        self.target_transform = lambda x: (x - self.means) / (self.stdevs +
                                                              1e-5)

    def __len__(self):
        return self.len_

    def __getitem__(self, idx):
        return self.getitem(idx)

    def getitem_nofinetune(self, idx):
        """
        Returns:
            data torch.Tensor([channels, h, w])
            label torch.Tensor([channels, h, w])
        """
        data = self.transform(torch.tensor(self.data[idx]))[:, 6:7]
        label = self.target_transform(torch.tensor(
            self.data[idx + self.tslag]))[:, 6:7]
        return data, label

    def getitem_finetune(self, idx):
        """
        Returns:
            data torch.Tensor([channels, h, w])
            label [torch.Tensor([channels, h, w]), torch.Tensor([channels, h, w])]
        """
        data = self.transform(torch.tensor(self.data[idx]))[:, 6:7]
        label = self.target_transform(torch.tensor(
            self.data[idx + self.tslag]))[:, 6:7], self.target_transform(
                torch.tensor(self.data[idx + 2 * self.tslag]))[:, 6:7]
        return data, label
