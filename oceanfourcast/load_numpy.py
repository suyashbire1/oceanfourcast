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
            ds.PHIHYD.isel(Zmd000015=-1), # Pbot
            ds.PsiVEL.sum('Zmd000015')    # Psi
        ]

        data = [channel.values.squeeze()[np.newaxis,...] for channel in channels]
        data[0] = u_corner_to_center(data[0])
        data[1] = u_corner_to_center(data[1])
        data[2] = v_corner_to_center(data[2])
        data[3] = v_corner_to_center(data[3])

        tau, Trest = create_forcing_arrays(xarray_data_file)
        data.append(tau)
        data.append(Trest)

        data = np.vstack(data)
        data = np.moveaxis(data, 0, 1)

        print("Calculating stats...")
        means = np.mean(data, axis=(0,2,3))
        stdevs = np.std(data, axis=(0,2,3))
        timemeans = np.mean(data, axis=0)
        timestdevs = np.std(data, axis=0)

        print("Saving data...")
        numpy_data_file = os.path.join(data_dir, "dynDiags.npy")
        np.save(numpy_data_file, data)
        numpy_stats_file = os.path.join(data_dir, "dynDiagsStats.npz")
        np.savez(numpy_stats_file, means=means, stdevs=stdevs, timemeans=timemeans, timestdevs=timestdevs)


def create_forcing_arrays(xarray_data_file):
    with xr.open_dataset(xarray_data_file, decode_times=False) as ds:
        x = ds.X.values
        y = ds.Y.values
    data_dir = os.path.dirname(xarray_data_file)
    forcing_file = os.path.join(data_dir, "forcing.json")
    with open(forcing_file) as f:
        forcing = json.load(f)
        Tmax = forcing['Tmax']
        Tmin = forcing['Tmin']
        taumax = forcing['taumax']
    nx, ny = x.size, y.size
    xo, yo = x[0], y[0]
    dx, dy = np.diff(x)[0], np.diff(y)[0]
    tau = -taumax * np.cos(2*np.pi*((y-yo)/(ny-2)/dy))
    tau = np.repeat(tau[:, np.newaxis], x.size, axis=1)
    Trest = (Tmax-Tmin)/(ny-2)/dy * (yo-y) + Tmax
    Trest = np.repeat(Trest[:, np.newaxis], x.size, axis=1)
    return tau, Trest

def u_corner_to_center(u):
    return (u[...,:-1] + u[...,1:])/2

def v_corner_to_center(v):
    return (v[...,:-1,:] + v[...,1:,:])/2

if __name__ == "__main__":
    save_numpy_file_from_xarray()


class OceanDataset(Dataset):
    def __init__(self, data_file, tslag=3, spinupts=0, fine_tune=False, device='cpu'):
        self.tslag = tslag
        self.spinupts = spinupts

        data_dir = os.path.dirname(data_file)

        mmap_mode = None
        if device == 'cpu':
            mmap_mode = 'r'

        self.data = np.load(data_file, mmap_mode=mmap_mode)[spinupts:]
        stats_file = np.load(os.path.join(data_dir, "dynDiagsStats.npz"))
        self.means = stats_file['means']
        self.stdevs = stats_file['stdevs']
        self.timemeans = stats_file['timemeans']
        self.timestdevs = stats_file['timestdevs']

        self.transform = transforms.Normalize(mean=self.means, std=self.stdevs)
        self.target_transform = transforms.Normalize(mean=self.means, std=self.stdevs)

        self.img_size = [self.data.shape[-1], self.data.shape[-2]]
        self.channels = self.data.shape[1]
        self.fine_tune = fine_tune
        if self.fine_tune:
            self.len_ = self.data.shape[0] - self.tslag*2
            self.__getitem__ = self.getitem_finetune
        else:
            self.len_ = self.data.shape[0] - self.tslag
            self.__getitem__ = self.getitem_nofinetune

    def __len__(self):
        return self.len_

    def getitem_nofinetune(self):
        """
        Returns:
            data torch.Tensor([channels, h, w])
            label torch.Tensor([channels, h, w])
        """
        data = self.transform(torch.tensor(self.data[idx]))
        label = self.target_transform(torch.tensor(self.data[idx+self.tslag]))
        return data, label

    def getitem_finetune(self):
        """
        Returns:
            data torch.Tensor([channels, h, w])
            label [torch.Tensor([channels, h, w]), torch.Tensor([channels, h, w])]
        """
        data = self.transform(torch.tensor(self.data[idx]))
        label = self.target_transform(torch.tensor(self.data[idx+self.tslag])), self.target_transform(torch.tensor(self.data[idx+2*self.tslag]))
        return data, label
