import os
import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np
import json
from torchvision import transforms

class OceanDataset(Dataset):
    def __init__(self, data_file, for_validate=False, tslag=3, spinupts=0):
        self.for_validate = for_validate
        self.tslag = tslag

        self.data_dir = os.path.dirname(data_file)
        self.ds = xr.open_dataset(data_file, decode_times=False)#, chunks=dict(T=10))
        self.img_size = [self.ds.X.size, self.ds.Y.size]
        self.spinupts = spinupts
        self.T_spinup = slice(spinupts,None)

        self.channels = [
                self.ds.UVEL.isel(Zmd000015=0),    # usurf
                self.ds.UVEL.isel(Zmd000015=7),    # umid
                self.ds.VVEL.isel(Zmd000015=0),    # vsurf
                self.ds.VVEL.isel(Zmd000015=7),    # vmid
                self.ds.THETA.isel(Zmd000015=0),   # Tsurf
                self.ds.THETA.isel(Zmd000015=7),   # Tmid
                self.ds.PHIHYD.isel(Zmd000015=0),  # Psurf
                self.ds.PHIHYD.isel(Zmd000015=7),  # Pmid
                self.ds.PHIHYD.isel(Zmd000015=-1)  # Pbot
                ]

        self.stats_file = os.path.join(self.data_dir, "stats.npz")
        if os.path.exists(self.stats_file):
            statsnpzfile = np.load(self.stats_file)
            self.means = statsnpzfile['means']
            self.stdevs = statsnpzfile['stdevs']
        else:
            self.calc_mean_std()

        self.transform = transforms.Normalize(mean=self.means, std=self.stdevs)
        self.target_transform = transforms.Normalize(mean=self.means, std=self.stdevs)

    def calc_mean_std(self):
        print('Calculating stats...')
        means = []
        stdevs = []
        for channel in self.channels:
            means.append(np.float32(channel.isel(T=slice(0,None)).mean().values))
            stdevs.append(np.float32(channel.isel(T=slice(0,None)).std().values))
        self.means = means
        self.stdevs = stdevs
        np.savez(self.stats_file, means=means, stdevs=stdevs)

    def close(self):
        self.ds.close()

    def __len__(self):
        n = len(self.ds.T.isel(T=self.T_spinup))
        if self.for_validate:
            return n // 10
        else:
            return n - n // 10

    @staticmethod
    def u_corner_to_center(u):
        return (u[...,:-1] + u[...,1:])/2

    @staticmethod
    def v_corner_to_center(v):
        return (v[...,:-1,:] + v[...,1:,:])/2

    def get_data(self, idx):
        data = [channel.isel(T=idx).values.squeeze()[np.newaxis,...] for channel in self.channels]
        data[0] = self.u_corner_to_center(data[0])
        data[1] = self.u_corner_to_center(data[1])
        data[2] = self.v_corner_to_center(data[2])
        data[3] = self.v_corner_to_center(data[3])
        data = np.vstack(data)
        return data

    def __getitem__(self, idx):
        idx = idx + self.spinupts
        n = len(self.ds.T.isel(T=self.T_spinup))

        if self.for_validate:
            idx = n - n //10 + idx - self.tslag

        data = self.get_data(idx)
        T    = self.get_data(idx + self.tslag)

        return data, T
