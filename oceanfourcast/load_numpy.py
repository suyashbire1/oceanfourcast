import os
import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np
import json
from torchvision import transforms
import click

@click.command()
@click.option("--xarray_data_file", default="./dynDiag.nc")
def save_numpy_file_from_xarray(xarray_data_file):
    data_dir = os.path.dirname(xarray_data_file)
    print("Loading data...")
    with xr.open_dataset(xarray_data_file, decode_times=False) as ds:
        channels = [
            ds.UVEL.isel(Zmd000015=0),    # usurf
            ds.UVEL.isel(Zmd000015=7),    # umid
            ds.VVEL.isel(Zmd000015=0),    # vsurf
            ds.VVEL.isel(Zmd000015=7),    # vmid
            ds.THETA.isel(Zmd000015=0),   # Tsurf
            ds.THETA.isel(Zmd000015=7),   # Tmid
            ds.PHIHYD.isel(Zmd000015=0),  # Psurf
            ds.PHIHYD.isel(Zmd000015=7),  # Pmid
            ds.PHIHYD.isel(Zmd000015=-1)  # Pbot
        ]

        data = [channel.values.squeeze()[np.newaxis,...] for channel in channels]
        data[0] = u_corner_to_center(data[0])
        data[1] = u_corner_to_center(data[1])
        data[2] = v_corner_to_center(data[2])
        data[3] = v_corner_to_center(data[3])
        data = np.vstack(data)
        data = np.moveaxis(data, 0, 1)

        print("Calculating stats...")
        means = np.mean(data, axis=(0,2,3))
        stdevs = np.std(data, axis=(0,2,3))

        print("Saving data...")
        numpy_data_file = os.path.join(data_dir, "dynDiags.npz")
        np.savez(numpy_data_file, means=means, stdevs=stdevs, data=data)

def u_corner_to_center(u):
    return (u[...,:-1] + u[...,1:])/2

def v_corner_to_center(v):
    return (v[...,:-1,:] + v[...,1:,:])/2

if __name__ == "__main__":
    save_numpy_file_from_xarray()


class OceanDataset(Dataset):
    def __init__(self, data_file, tslag=3, spinupts=0):
        self.tslag = tslag
        self.spinupts = spinupts

        data_file = np.load(data_file)
        self.data = data_file['data'][spinupts:]
        self.means = data_file['means']
        self.stdevs = data_file['stdevs']

        self.transform = transforms.Normalize(mean=self.means, std=self.stdevs)
        self.target_transform = transforms.Normalize(mean=self.means, std=self.stdevs)

        self.img_size = [self.data.shape[-1], self.data.shape[-2]]
        self.channels = self.data.shape[1]

    def __len__(self):
        return self.data.shape[0] - self.tslag

    def __getitem__(self, idx):
        data = torch.tensor(self.data[idx])
        label = torch.tensor(self.data[idx+self.tslag])
        return self.transform(data), self.target_transform(label)
