import os
import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np

class OceanDataset(Dataset):
    def __init__(self, data_file, transform=None, target_transform=None, for_validate=False, tslag=3, spinupts=0):
        self.transform = transform
        self.target_transform = target_transform
        self.for_validate = for_validate
        self.tslag = tslag

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

        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            T = self.target_transform(T)

        return data, T
