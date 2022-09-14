import os
import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np

class OceanDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_transform=None, for_validate=False, tslag=3, spinupts=0, normalize=False):
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.ncfile = os.path.join(self.data_dir + "dynDiag.nc")
        self.for_validate = for_validate
        self.tslag = tslag

        ds = xr.open_dataset(self.ncfile, decode_times=False)
        self.ds = ds
        self.img_size = [ds.X.size, ds.Y.size]
        self.spinupts = spinupts
        self.T_spinup = slice(spinupts,None)

        if normalize:
            self.usurfmean = ds.UVEL.isel(Zmd000015=0, T=self.T_spinup).mean().values
            self.usurfstd = ds.UVEL.isel(Zmd000015=0, T=self.T_spinup).std().values
            self.umidmean = ds.UVEL.isel(Zmd000015=7, T=self.T_spinup).mean().values
            self.umidstd = ds.UVEL.isel(Zmd000015=7, T=self.T_spinup).std().values

            self.vsurfmean = ds.VVEL.isel(Zmd000015=0, T=self.T_spinup).mean().values
            self.vsurfstd = ds.VVEL.isel(Zmd000015=0, T=self.T_spinup).std().values
            self.vmidmean = ds.VVEL.isel(Zmd000015=7, T=self.T_spinup).mean().values
            self.vmidstd = ds.VVEL.isel(Zmd000015=7, T=self.T_spinup).std().values

            # self.wmidmean = ds.WVEL.isel(Zld000015=7, T=self.T_spinup).mean().values
            # self.wmidstd = ds.WVEL.isel(Zld000015=7, T=self.T_spinup).std().values

            self.thetasurfmean = ds.THETA.isel(Zmd000015=0, T=self.T_spinup).mean().values
            self.thetasurfstd = ds.THETA.isel(Zmd000015=0, T=self.T_spinup).std().values
            self.thetamidmean = ds.THETA.isel(Zmd000015=7, T=self.T_spinup).mean().values
            self.thetamidstd = ds.THETA.isel(Zmd000015=7, T=self.T_spinup).std().values

            self.psurfmean = ds.PHIHYD.isel(Zmd000015=0, T=self.T_spinup).mean().values
            self.psurfstd = ds.PHIHYD.isel(Zmd000015=0, T=self.T_spinup).std().values
            self.pmidmean = ds.PHIHYD.isel(Zmd000015=7, T=self.T_spinup).mean().values
            self.pmidstd = ds.PHIHYD.isel(Zmd000015=7, T=self.T_spinup).std().values
            self.pbotmean = ds.PHIHYD.isel(Zmd000015=-1, T=self.T_spinup).mean().values
            self.pbotstd = ds.PHIHYD.isel(Zmd000015=-1, T=self.T_spinup).std().values
        else:
            self.usurfmean = 0.0
            self.usurfstd = 1.0
            self.umidmean = 0.0
            self.umidstd = 1.0

            self.vsurfmean = 0.0
            self.vsurfstd = 1.0
            self.vmidmean = 0.0
            self.vmidstd = 1.0

            # self.wmidmean = 0.0
            # self.wmidstd = 1.0

            self.thetasurfmean = 0.0
            self.thetasurfstd = 1.0
            self.thetamidmean = 0.0
            self.thetamidstd = 1.0

            self.psurfmean = 0.0
            self.psurfstd = 1.0
            self.pmidmean = 0.0
            self.pmidstd = 1.0
            self.pbotmean = 0.0
            self.pbotstd = 1.0

    def close(self):
        self.ds.close()

    def __len__(self):
        n = len(self.ds.T.isel(T=self.T_spinup))
        if self.for_validate:
            return n // 10
        else:
            return n - n // 10

    def get_data(self, idx):
        usurf = (self.ds.UVEL.isel(T=idx, Zmd000015=0).values.squeeze() - self.usurfmean)/self.usurfstd
        usurf = (usurf[...,:-1] + usurf[...,1:])/2
        umid = (self.ds.UVEL.isel(T=idx, Zmd000015=7).values.squeeze() - self.umidmean)/self.umidstd
        umid = (umid[...,:-1] + umid[...,1:])/2

        vsurf = (self.ds.VVEL.isel(T=idx, Zmd000015=0).values.squeeze() - self.vsurfmean)/self.vsurfstd
        vsurf = (vsurf[...,:-1,:] + vsurf[...,1:,:])/2
        vmid = (self.ds.VVEL.isel(T=idx, Zmd000015=7).values.squeeze() - self.vmidmean)/self.vmidstd
        vmid = (vmid[...,:-1,:] + vmid[...,1:,:])/2

        # wmid = (self.ds.WVEL.isel(T=idx, Zld000015=7).values.squeeze() - self.wmidmean)/self.wmidstd

        thetasurf = (self.ds.THETA.isel(T=idx, Zmd000015=0).values.squeeze() - self.thetasurfmean)/self.thetasurfstd
        thetamid = (self.ds.THETA.isel(T=idx, Zmd000015=7).values.squeeze() - self.thetamidmean)/self.thetamidstd

        Psurf = (self.ds.PHIHYD.isel(T=idx, Zmd000015=0).values.squeeze() - self.psurfmean)/self.psurfstd
        Pmid = (self.ds.PHIHYD.isel(T=idx, Zmd000015=7).values.squeeze() - self.pmidmean)/self.pmidstd
        Pbot = (self.ds.PHIHYD.isel(T=idx, Zmd000015=-1).values.squeeze() - self.pbotmean)/self.pbotstd

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
