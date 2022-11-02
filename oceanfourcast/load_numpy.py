import os
import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np
import json
from torchvision import transforms

def save_numpy_file_from_xarray(xarray_data_file):
    data_dir = os.path.dirname(xarray_data_file)
    print("Loading data...")
    with xr.open_dataset(xarray_data_file, decode_times=False) as ds:
        Tslice = slice(0, 10)
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

        data = [channel.isel(T=Tslice).values.squeeze()[np.newaxis,...] for channel in self.channels]
        data[0] = u_corner_to_center(data[0])
        data[1] = u_corner_to_center(data[1])
        data[2] = v_corner_to_center(data[2])
        data[3] = v_corner_to_center(data[3])
        data = np.vstack(data)

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
