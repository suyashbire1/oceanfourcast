import os
import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np

class OceanDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.ncfile = os.path.join(self.data_dir + "dynDiag.nc")

    def __len__(self):
        with xr.open_dataset(self.ncfile, decode_times=False) as ds:
            return len(ds.T)

    def __getitem__(self, idx):
        with xr.open_dataset(self.ncfile, decode_times=False) as ds:
            print('here')
            usurf = ds.UVEL.isel(T=idx, Zmd000015=0).values.squeeze()
            print('here2')
            usurf = (usurf[...,:-1] + usurf[...,1:])/2
            print('here3')
            vsurf = ds.VVEL.isel(T=idx, Zmd000015=0).values.squeeze()
            vsurf = (vsurf[...,:-1,:] + vsurf[...,1:,:])/2
            wmid = ds.WVEL.isel(T=idx, Zld000015=7).values.squeeze()
            thetasurf = ds.THETA.isel(T=idx, Zmd000015=0).values.squeeze()
            Psurf = ds.PHIHYD.isel(T=idx, Zmd000015=0).values.squeeze()
            Pmid = ds.PHIHYD.isel(T=idx, Zmd000015=7).values.squeeze()
            channels = [usurf, vsurf, wmid, thetasurf, Psurf, Pmid]
            data = np.vstack([channel[np.newaxis,...] for channel in channels])
            T = ds.T.isel(T=idx).values
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            T = self.target_transform(T)
        return data, T
