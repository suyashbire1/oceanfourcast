import os
import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np

class OceanDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_transform=None, for_validate=False):
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.ncfile = os.path.join(self.data_dir + "dynDiag.nc")
        self.for_validate = for_validate
        self.ds = xr.open_dataset(self.ncfile, decode_times=False)
        self.img_size = [ds.X.size, ds.Y.size]

    def close(self):
        self.ds.close()

    def __len__(self):
        n = len(self.ds.T)
        if self.for_validate:
            return n // 10
        else:
            return n - n // 10

    def __getitem__(self, idx):
        n = len(self.ds.T)
        if self.for_validate:
            idx = n - n //10 + idx
        usurf = self.ds.UVEL.isel(T=idx, Zmd000015=0).values.squeeze()
        usurf = (usurf[...,:-1] + usurf[...,1:])/2
        vsurf = self.ds.VVEL.isel(T=idx, Zmd000015=0).values.squeeze()
        vsurf = (vsurf[...,:-1,:] + vsurf[...,1:,:])/2
        wmid = self.ds.WVEL.isel(T=idx, Zld000015=7).values.squeeze()
        thetasurf = self.ds.THETA.isel(T=idx, Zmd000015=0).values.squeeze()
        Psurf = self.ds.PHIHYD.isel(T=idx, Zmd000015=0).values.squeeze()
        Pmid = self.ds.PHIHYD.isel(T=idx, Zmd000015=7).values.squeeze()
        channels = [usurf, vsurf, wmid, thetasurf, Psurf, Pmid]
        data = np.vstack([channel[np.newaxis,...] for channel in channels])
        T = self.ds.T.isel(T=idx).values

        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            T = self.target_transform(T)
        return data, T
