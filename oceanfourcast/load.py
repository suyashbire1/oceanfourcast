import os
import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np

class OceanDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_transform=None, for_validate=False, tslag=3, spinupts=0):
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.ncfile = os.path.join(self.data_dir + "dynDiag.nc")
        self.for_validate = for_validate
        self.tslag = tslag

        ds = xr.open_dataset(self.ncfile, decode_times=False)#, chunks=dict(T=10))
        self.ds = ds
        self.img_size = [ds.X.size, ds.Y.size]
        self.spinupts = spinupts
        self.T_spinup = slice(spinupts,None)

    def close(self):
        self.ds.close()

    def __len__(self):
        n = len(self.ds.T.isel(T=self.T_spinup))
        if self.for_validate:
            return n // 10
        else:
            return n - n // 10

    def get_data(self, idx):
        usurf = self.ds.UVEL.isel(T=idx, Zmd000015=0).values.squeeze()
        usurf = (usurf[...,:-1] + usurf[...,1:])/2
        umid = self.ds.UVEL.isel(T=idx, Zmd000015=7).values.squeeze()
        umid = (umid[...,:-1] + umid[...,1:])/2

        vsurf = self.ds.VVEL.isel(T=idx, Zmd000015=0).values.squeeze()
        vsurf = (vsurf[...,:-1,:] + vsurf[...,1:,:])/2
        vmid = self.ds.VVEL.isel(T=idx, Zmd000015=7).values.squeeze()
        vmid = (vmid[...,:-1,:] + vmid[...,1:,:])/2

        # wmid = self.ds.WVEL.isel(T=idx, Zld000015=7).values.squeeze()

        thetasurf = self.ds.THETA.isel(T=idx, Zmd000015=0).values.squeeze()
        thetamid = self.ds.THETA.isel(T=idx, Zmd000015=7).values.squeeze()

        Psurf = self.ds.PHIHYD.isel(T=idx, Zmd000015=0).values.squeeze()
        Pmid = self.ds.PHIHYD.isel(T=idx, Zmd000015=7).values.squeeze()
        Pbot = self.ds.PHIHYD.isel(T=idx, Zmd000015=-1).values.squeeze()

        # channels = [usurf, umid, vsurf, vmid, wmid, thetasurf, thetamid, Psurf, Pmid, Pbot]
        channels = [usurf, umid, vsurf, vmid, thetasurf, thetamid, Psurf, Pmid, Pbot]
        data = np.vstack([channel[np.newaxis,...] for channel in channels])
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
