import os
import torch
from torch.utils.data import Dataset
import netCDF4
import numpy as np

class OceanDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_transform=None, for_validate=False, tslag=3, spinupts=0):
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.ncfile = os.path.join(self.data_dir + "dynDiag.nc")
        self.for_validate = for_validate
        self.tslag = tslag

        ds = netCDF4.Dataset(self.ncfile, 'r')
        self.ds = ds
        self.img_size = [ds['X'].size, ds['Y'].size]
        self.spinupts = spinupts
        self.T_spinup = slice(spinupts,None)

    def close(self):
        self.ds.close()

    def __len__(self):
        n = len(self.ds['T'][self.T_spinup])
        if self.for_validate:
            return n // 10
        else:
            return n - n // 10

    def get_data(self, idx):
        usurf = self.ds['UVEL'][idx, 0].squeeze()
        usurf = (usurf[...,:-1] + usurf[...,1:])/2
        umid = self.ds['UVEL'][idx, 7].squeeze()
        umid = (umid[...,:-1] + umid[...,1:])/2

        vsurf = self.ds['VVEL'][idx,0].squeeze()
        vsurf = (vsurf[...,:-1,:] + vsurf[...,1:,:])/2
        vmid = self.ds['VVEL'][idx, 7].squeeze()
        vmid = (vmid[...,:-1,:] + vmid[...,1:,:])/2

        # wmid = self.ds['WVEL'][idx,7].squeeze()

        thetasurf = self.ds['THETA'][idx,0].squeeze()
        thetamid = self.ds['THETA'][idx, 7].squeeze()

        Psurf = self.ds['PHIHYD'][idx,0].squeeze()
        Pmid = self.ds['PHIHYD'][idx, 7].squeeze()
        Pbot = self.ds['PHIHYD'][idx, -1].squeeze()

        # channels = [usurf, umid, vsurf, vmid, wmid, thetasurf, thetamid, Psurf, Pmid, Pbot]
        channels = [usurf, umid, vsurf, vmid, thetasurf, thetamid, Psurf, Pmid, Pbot]
        data = np.vstack([channel[np.newaxis,...] for channel in channels])
        return data

    def __getitem__(self, idx):
        idx = idx + self.spinupts
        n = len(self.ds['T'][self.T_spinup])

        if self.for_validate:
            idx = n - n //10 + idx - self.tslag

        data = self.get_data(idx)
        T    = self.get_data(idx + self.tslag)

        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            T = self.target_transform(T)

        return data.filled(), T.filled()
