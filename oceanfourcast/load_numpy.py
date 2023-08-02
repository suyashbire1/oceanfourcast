import os
import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np
import json
from torchvision import transforms
import click
import glob
from neuralop.datasets.transforms import PositionalEmbedding, Normalizer


@click.command()
@click.option("--xarray_data_file", default="./dynDiag.nc")
def save_numpy_file_from_xarray(xarray_data_file):
    data_dir = os.path.dirname(xarray_data_file)
    print("Loading data...")
    with xr.open_dataset(xarray_data_file, decode_times=False) as ds:
        channels = [
            ds.UVEL.isel(Zmd000015=0),  # usurf
            ds.UVEL.isel(Zmd000015=7),  # umid
            ds.VVEL.isel(Zmd000015=0),  # vsurf
            ds.VVEL.isel(Zmd000015=7),  # vmid
            ds.THETA.isel(Zmd000015=0),  # Tsurf
            ds.THETA.isel(Zmd000015=7),  # Tmid
            ds.PHIHYD.isel(Zmd000015=0),  # Psurf
            ds.PHIHYD.isel(Zmd000015=7),  # Pmid
            ds.PHIHYD.isel(Zmd000015=-1),  # Pbot
            ds.PsiVEL.sum('Zmd000015')  # Psi
        ]

        data = [
            channel.values.squeeze()[np.newaxis, ...] for channel in channels
        ]
        data[0] = u_corner_to_center(data[0])
        data[1] = u_corner_to_center(data[1])
        data[2] = v_corner_to_center(data[2])
        data[3] = v_corner_to_center(data[3])
        data[9] = u_corner_to_center(v_corner_to_center(data[9]))

        nt = np.shape(data[0])[1]
        tau, Trest = create_forcing_arrays(xarray_data_file)
        tau = np.repeat(tau[np.newaxis, np.newaxis, :, :], nt, axis=1)
        Trest = np.repeat(Trest[np.newaxis, np.newaxis, :, :], nt, axis=1)
        data.append(tau)
        data.append(Trest)

        data = np.vstack(data)
        data = np.moveaxis(data, 0, 1)

        print("Calculating stats...")
        means = np.mean(data, axis=(0, 2, 3))
        stdevs = np.std(data, axis=(0, 2, 3))
        timemeans = np.mean(data, axis=0)
        timestdevs = np.std(data, axis=0)

        print("removing old files...")
        os.remove(xarray_data_file)

        print("Saving data...")
        numpy_data_file = os.path.join(data_dir, "dynDiags.npy")
        np.save(numpy_data_file, data)
        numpy_stats_file = os.path.join(data_dir, "dynDiagsStats.npz")
        np.savez(numpy_stats_file,
                 means=means,
                 stdevs=stdevs,
                 timemeans=timemeans,
                 timestdevs=timestdevs)


@click.command()
@click.option("--data_rootdir", default="./")
def save_global_stats(data_rootdir):
    file_pattern = os.path.join(data_rootdir, "**", "dynDiags.npy")
    files = glob.iglob(file_pattern, recursive=True)
    f = next(files)
    print(f"Working on {f}")
    data = np.load(f, mmap_mode='r')
    nt = data.shape[0]
    global_means = nt * np.mean(data, axis=(0, 2, 3))
    global_stdevs = nt * np.std(data, axis=(0, 2, 3))
    global_time_steps = nt

    for f in files:
        print(f"Working on {f}")
        data = np.load(f, mmap_mode='r')
        nt = data.shape[0]
        sim_means = np.mean(data, axis=(0, 2, 3))
        sim_stdevs = np.std(data, axis=(0, 2, 3))
        global_means += nt * sim_means
        global_stdevs += nt * sim_stdevs
        global_time_steps += nt

    global_means /= global_time_steps
    global_stdevs /= global_time_steps
    print(global_means)
    print(global_stdevs)

    print("Saving data...")
    file_pattern = os.path.join(data_rootdir, "**", "dynDiags.npy")
    files = glob.iglob(file_pattern, recursive=True)
    for f in files:
        data_dir = os.path.dirname(f)
        global_stats_file = os.path.join(data_dir, "dynDiagsGlobalStats.npz")
        np.savez(global_stats_file, means=global_means, stdevs=global_stdevs)


@click.command()
@click.option("--data_rootdir", default="./")
def save_global_stats_wind_only(data_rootdir):
    file_pattern = os.path.join(data_rootdir, "**", "dynDiags.npy")
    files = glob.iglob(file_pattern, recursive=True)
    f = next(files)
    print(f"Working on {f}")
    data = np.load(f, mmap_mode='r')
    nt = data.shape[0]
    global_timemeans = nt * np.mean(data, axis=0)
    global_time_steps = nt

    for f in files:
        print(f"Working on {f}")
        data = np.load(f, mmap_mode='r')
        nt = data.shape[0]
        global_timemeans += nt * np.mean(data, axis=0)
        global_time_steps += nt

    global_timemeans /= global_time_steps

    file_pattern = os.path.join(data_rootdir, "**", "dynDiags.npy")
    files = glob.iglob(file_pattern, recursive=True)
    f = next(files)
    print(f"Working on {f}")
    data = np.load(f, mmap_mode='r')
    nt = data.shape[0]
    global_var = (data[0] - global_timemeans)**2
    for i in range(1, nt):
        global_var += (data[i] - global_timemeans)**2
    for f in files:
        print(f"Working on {f}")
        data = np.load(f, mmap_mode='r')
        nt = data.shape[0]
        for i in range(nt):
            global_var += (data[i] - global_timemeans)**2

    global_timestdevs = np.sqrt(global_var / global_time_steps)

    print("Saving data...")
    file_pattern = os.path.join(data_rootdir, "**", "dynDiags.npy")
    files = glob.iglob(file_pattern, recursive=True)
    for f in files:
        data_dir = os.path.dirname(f)
        global_stats_file = os.path.join(data_dir, "dynDiagsGlobalStats2D.npz")
        np.savez(global_stats_file,
                 timemeans=global_timemeans,
                 timestdevs=global_timestdevs)


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
    tau = -taumax * np.cos(2 * np.pi * ((y - yo) / (ny - 2) / dy))
    tau = np.repeat(tau[:, np.newaxis], x.size, axis=1)
    Trest = (Tmax - Tmin) / (ny - 2) / dy * (yo - y) + Tmax
    Trest = np.repeat(Trest[:, np.newaxis], x.size, axis=1)
    return tau, Trest


def u_corner_to_center(u):
    return (u[..., :-1] + u[..., 1:]) / 2


def v_corner_to_center(v):
    return (v[..., :-1, :] + v[..., 1:, :]) / 2


if __name__ == "__main__":
    #save_numpy_file_from_xarray()
    save_global_stats_wind_only()


class OceanDataset(Dataset):

    def __init__(self,
                 data_file,
                 tslag=3,
                 spinupts=0,
                 fine_tune=False,
                 device='cpu',
                 multi_expt_normalize=False,
                 mmap_mode=None):
        self.tslag = tslag
        self.spinupts = spinupts

        data_dir = os.path.dirname(data_file)

        if device == 'cpu':
            mmap_mode = 'r'

        self.data = np.load(data_file, mmap_mode=mmap_mode)[spinupts:]
        if multi_expt_normalize is True:
            stats_file = np.load(
                os.path.join(data_dir, "dynDiagsGlobalStats2D.npz"))
        else:
            stats_file = np.load(os.path.join(data_dir, "dynDiagsStats.npz"))
        self.means = stats_file['timemeans']
        self.stdevs = stats_file['timestdevs']

        self.img_size = [self.data.shape[-1], self.data.shape[-2]]
        h, w = self.img_size
        self.channels = self.data.shape[1]
        self.fine_tune = fine_tune
        self.means = np.concatenate(
            (self.means, np.zeros(
                (1, self.data.shape[-1], self.data.shape[-2]))),
            axis=0)
        self.stdevs = np.concatenate(
            (self.stdevs, np.ones(
                (1, self.data.shape[-1], self.data.shape[-2]))),
            axis=0)
        if self.fine_tune:
            self.len_ = self.data.shape[0] - self.tslag * 2
            self.getitem = self.getitem_finetune
        else:
            self.len_ = self.data.shape[0] - self.tslag
            self.getitem = self.getitem_nofinetune

        # self.transform = transforms.Normalize(mean=self.means, std=self.stdevs)
        # self.target_transform = transforms.Normalize(mean=self.means,
        self.transform = lambda x: (x - self.means) / (self.stdevs + 1e-5)
        self.pos_embed = lambda x: torch.cat(
            x, np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w)), dim=0)
        self.target_transform = lambda x: (x - self.means) / (self.stdevs +
                                                              1e-5)

    def __len__(self):
        return self.len_

    def __getitem__(self, idx):
        return self.getitem(idx)

    def getitem_nofinetune(self, idx):
        """
        Returns:
            data torch.Tensor([channels, h, w])
            label torch.Tensor([channels, h, w])
        """
        data = self.pos_embed(self.transform(torch.tensor(self.data[idx])))
        label = self.target_transform(torch.tensor(self.data[idx +
                                                             self.tslag]))
        return data, label

    def getitem_finetune(self, idx):
        """
        Returns:
            data torch.Tensor([channels, h, w])
            label [torch.Tensor([channels, h, w]), torch.Tensor([channels, h, w])]
        """
        data = self.pos_embed(self.transform(torch.tensor(self.data[idx])))
        label = self.target_transform(torch.tensor(
            self.data[idx + self.tslag])), self.target_transform(
                torch.tensor(self.data[idx + 2 * self.tslag]))
        return data, label
