import os
import sys
import glob
import matplotlib.pyplot as plt
import json
import numpy as np
import torch
import xarray as xr
from collections import defaultdict
from oceanfourcast import load_numpy as load, fourcastnet, unet
from neuralop.models import FNO
import importlib

importlib.reload(load)


class Experiment():

    def __init__(self, expt_dir):
        self.expt_dir = expt_dir
        self.name = expt_dir.rsplit('/', 1)[1]
        with open(os.path.join(expt_dir, "logfile.json"), 'r') as f:
            logs = json.load(f)
        for k, v in logs.items():
            setattr(self, k, v)

    def plot_train_loss(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        ax.plot(self.training_loss, label=self.name)
        ax.set_xlabel('Minibatch')
        ax.set_ylabel('Train Loss')
        return ax.get_figure()

    def plot_train_valid_loss(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        ax.plot(self.avg_training_loss, label=self.name + ' train loss')
        ax.plot(self.validation_loss, label=self.name + ' valid loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.grid()
        return ax.get_figure()

    def recreate_model(self, epoch=None, device='cpu'):
        if self.modelstr == 'unet':
            self.model = unet.UNet(n_channels=self.in_channels,
                                   n_classes=self.out_channels)
        elif self.modelstr == 'fno':
            self.model = FNO(n_modes=(self.nfmodes, self.nfmodes),
                             n_layers=self.depth,
                             hidden_channels=self.embed_dims,
                             in_channels=self.in_channels,
                             out_channels=self.out_channels,
                             device=device).to(device)
        else:
            self.model = fourcastnet.AFNONet(
                embed_dim=self.embed_dims,
                patch_size=self.patch_size,
                sparsity=self.sparsity,
                img_size=[self.image_height, self.image_width],
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                drop_rate=self.drop_rate)

        if epoch is None:
            epoch = self.best_vloss_epoch
        model_path = os.path.join(self.expt_dir, f'model_epoch_{epoch}')
        self.model.load_state_dict(
            torch.load(model_path, map_location=torch.device(device)))

    def truth_compare_one_timestep(self,
                                   data_file=None,
                                   timestep=30,
                                   labels=None,
                                   cmaps=None,
                                   device='cpu'):
        model = self.model
        if data_file is None:
            data_file = self.data_file
        ds = load.OceanDataset(data_file,
                               spinupts=self.spinupts,
                               tslag=self.tslag,
                               multi_expt_normalize=True,
                               device=device)

        with torch.no_grad():
            yi, yip1 = ds[timestep]  # yi, yi + tau
            yi = yi.unsqueeze(0).to(device, dtype=torch.float)
            yip1 = yip1.unsqueeze(0).to(device, dtype=torch.float)
            yip1hat = model(yi)  # fourcastnet predicted yi + tau

            # yip1 = yip1*ds.stdevs[:,np.newaxis,np.newaxis] + ds.means[:,np.newaxis,np.newaxis]
            # yip1 = yip1*ds.stdevs[:,np.newaxis,np.newaxis] + ds.means[:,np.newaxis,np.newaxis]
            # yip1hat = yip1hat*ds.stdevs[:,np.newaxis,np.newaxis] + ds.means[:,np.newaxis,np.newaxis]

        if cmaps is None:
            cmaps = [
                'RdBu_r', 'RdBu_r', 'RdBu_r', 'RdBu_r', 'RdYlBu_r', 'RdYlBu_r',
                'RdYlBu_r', 'RdYlBu_r', 'RdYlBu_r', 'RdBu_r'
            ]
        if labels is None:
            labels = [
                'U_surf', 'umid', 'V_surf', 'vmid', 'T_surf', 'thetamid',
                'P_surf', 'pmid', 'pbot', 'Psi'
            ]
        with plt.style.context(('labelsize15')):
            fig, ax = plt.subplots(10,
                                   3,
                                   sharex=True,
                                   sharey=True,
                                   figsize=(15, 35))
            lon, lat = ds.img_size
            lon = np.linspace(0, 62, lon)
            lat = np.linspace(10, 72, lat)
            for i in range(self.out_channels):
                if cmaps[i] == 'RdBu_r':
                    vmax = np.nanpercentile(np.fabs(yip1.squeeze()), 99)
                    vmin = -vmax
                else:
                    vmin, vmax = np.nanpercentile(yip1.squeeze(), (1, 99))
                im = ax[i, 0].pcolormesh(lon,
                                         lat,
                                         yi.squeeze()[i],
                                         vmin=vmin,
                                         vmax=vmax,
                                         cmap=cmaps[i])
                fig.colorbar(im, ax=ax[i, 0])
                im = ax[i, 1].pcolormesh(lon,
                                         lat,
                                         yip1.squeeze()[i],
                                         vmin=vmin,
                                         vmax=vmax,
                                         cmap=cmaps[i])
                fig.colorbar(im, ax=ax[i, 1])
                im = ax[i, 2].pcolormesh(lon,
                                         lat,
                                         yip1hat.squeeze()[i],
                                         vmin=vmin,
                                         vmax=vmax,
                                         cmap=cmaps[i])
                fig.colorbar(im, ax=ax[i, 2])
                ax[i, 0].set_title(f'{labels[i]}, Initial ($t=0$)')
                ax[i, 1].set_title(f'{labels[i]}, Truth ($t=\Delta T$)')
                ax[i, 2].set_title(f'{labels[i]}, FourCastNet ($t=\Delta T$)')

            for axc in ax[-1, :]:
                axc.set_xlabel(r'Lon ($^{\circ}$)')
            for axc in ax[:, 0]:
                axc.set_ylabel(r'Lat ($^{\circ}$)')
            for axc in ax.ravel():
                axc.set_aspect(1)
                for xp in lon[::4]:
                    axc.axvline(xp, c='k', lw=0.1)
                for yp in lat[::4]:
                    axc.axhline(yp, c='k', lw=0.1)

            fig.tight_layout()
            #fig.colorbar(im, ax=ax.ravel(), shrink=0.3)
        return fig

    def rmse_lw(self,
                n_samples=100,
                len_=100,
                data_file=None,
                device='cpu',
                channel=9):
        model = self.model
        if data_file is None:
            data_file = self.data_file
        ds = load.OceanDataset(data_file,
                               spinupts=self.spinupts,
                               tslag=self.tslag,
                               multi_expt_normalize=True,
                               device=device)
        ntime = len(ds) // 2
        nlon, nlat = ds.img_size
        lat = np.linspace(10, 72, nlat)
        latrad = np.deg2rad(lat)
        lw = nlat * np.cos(latrad) / np.sum(np.cos(latrad))
        lw = lw[:, np.newaxis]

        data_dir = os.path.dirname(data_file)
        stats_file = np.load(
            os.path.join(data_dir, "dynDiagsGlobalStats2D.npz"))
        means = stats_file['timemeans']
        means = means[:self.out_channels]
        stdevs = stats_file['timestdevs']
        stdevs = stdevs[:self.out_channels]

        ni = np.random.choice(ntime, n_samples)
        rmse_array = []
        for j in ni:
            sys.stdout.write(
                f"\rCreating correlation for time series beginnning at {j}...")
            rmses = []
            with torch.no_grad():
                ynext = ds[j][0].unsqueeze(0).to(device,
                                                 dtype=torch.float)  # yi
                for i, n in enumerate(range(j, j + len_, self.tslag)):
                    yip1 = ds[n][1].unsqueeze(0).to(
                        device, dtype=torch.float)  # yi + tau
                    yip1hat = model(ynext)
                    truth = yip1[:, :self.out_channels].detach().numpy(
                    ).squeeze()
                    pred = yip1hat.detach().numpy().squeeze()
                    truth = (stdevs * truth + means)
                    pred = (stdevs * pred + means)
                    truth = truth[channel]
                    pred = pred[channel]
                    rmse = np.sqrt(
                        np.sum(lw * (truth - pred)**2, axis=(1, 2)) / nlat /
                        nlon)
                    rmses.append(rmse)
                    ynext = torch.cat((yip1hat, yip1[:, self.out_channels:]),
                                      dim=1)
            rmse_array.append(rmses)
            sys.stdout.write(f"\r{j} done!")
        return np.array(rmse_array)

    def anomaly_correlation_coefficient_lw(self,
                                           n_samples=100,
                                           len_=100,
                                           data_file=None,
                                           device='cpu',
                                           channel=9):
        model = self.model
        if data_file is None:
            data_file = self.data_file
        ds = load.OceanDataset(data_file,
                               spinupts=self.spinupts,
                               tslag=self.tslag,
                               multi_expt_normalize=True,
                               device=device)
        ntime = len(ds) // 2
        _, nlat = ds.img_size
        lat = np.linspace(10, 72, nlat)
        latrad = np.deg2rad(lat)
        lw = nlat * np.cos(latrad) / np.sum(np.cos(latrad))
        lw = lw[:, np.newaxis]

        data_dir = os.path.dirname(data_file)
        # stats_file = np.load(os.path.join(data_dir, "dynDiagsStats.npz"))
        # timemeans = stats_file['timemeans']
        # timemeans = timemeans[:self.out_channels]
        # stats_file = np.load(os.path.join(data_dir, "dynDiagsGlobalStats.npz"))
        # means = stats_file['means'][:, np.newaxis, np.newaxis]
        # means = means[:self.out_channels]
        # stdevs = stats_file['stdevs'][:, np.newaxis, np.newaxis]
        # stdevs = stdevs[:self.out_channels]

        ni = np.random.choice(ntime, n_samples)
        acc_array = []
        for j in ni:
            sys.stdout.write(
                f"\rCreating correlation for time series beginnning at {j}...")
            accs = []
            with torch.no_grad():
                ynext = ds[j][0].unsqueeze(0).to(device,
                                                 dtype=torch.float)  # yi
                for i, n in enumerate(range(j, j + len_, self.tslag)):
                    yip1 = ds[n][1].unsqueeze(0).to(
                        device, dtype=torch.float)  # yi + tau
                    yip1hat = model(ynext)
                    truth = yip1[:, :self.out_channels].detach().numpy(
                    ).squeeze()
                    pred = yip1hat.detach().numpy().squeeze()
                    #truthanom = (stdevs * truth + means) - timemeans
                    #predanom = (stdevs * pred + means) - timemeans
                    truthanom = truth
                    predanom = pred
                    truthanom = truthanom[channel]
                    predanom = predanom[channel]
                    acc = np.sum(truthanom * predanom * lw,
                                 axis=(1, 2)) / np.sqrt(
                                     np.sum(lw * truthanom**2, axis=(1, 2)) *
                                     np.sum(lw * predanom**2, axis=(1, 2)))
                    accs.append(acc)
                    ynext = torch.cat((yip1hat, yip1[:, self.out_channels:]),
                                      dim=1)
            acc_array.append(accs)
            sys.stdout.write(f"\r{j} done!")
        return np.array(acc_array)

    def truth_pred_difference(self, ni, nf, data_file=None, device='cpu'):
        model = self.model
        if data_file is None:
            data_file = self.data_file
        ds = load.OceanDataset(data_file,
                               spinupts=self.spinupts,
                               tslag=self.tslag,
                               device=device)

        lon, lat = ds.img_size
        lon = np.linspace(0, 62, lon)
        lat = np.linspace(10, 72, lat)

        _, nlat = ds.img_size
        lat = np.linspace(10, 72, nlat)
        latrad = np.deg2rad(lat)
        lw = nlat * np.cos(latrad) / np.sum(np.cos(latrad))
        lw = lw[:, np.newaxis]

        data_dir = os.path.dirname(data_file)
        stats_file = np.load(os.path.join(data_dir, "dynDiagsStats.npz"))
        timemeans = stats_file['timemeans']
        timemeans = timemeans[:self.out_channels]

        accs = []
        with torch.no_grad():
            ynext = ds[ni][0].unsqueeze(0).to(device, dtype=torch.float)  # yi
            for i, n in enumerate(range(ni, nf)):
                yip1 = ds[n][1].unsqueeze(0).to(device,
                                                dtype=torch.float)  # yi + tau
                yip1hat = model(ynext)
                truth = yip1[:, :self.out_channels].detach().numpy().squeeze()
                pred = yip1hat.detach().numpy().squeeze()
                difference = truth - pred
                accs.append(difference)
                ynext = torch.cat((yip1hat, yip1[:, self.out_channels:]),
                                  dim=1)
        return accs, lon, lat


def create_experiments_dict(root_dir, pretrain=True):
    if pretrain:
        logfile_str = "pretrain/logfile.json"
    else:
        logfile_str = "finetune/logfile.json"
    logfile_pattern = os.path.join(root_dir, "**", logfile_str)
    files = glob.iglob(logfile_pattern, recursive=True)
    expts = []
    for f in files:
        expts.append(Experiment(os.path.dirname(f)))

    dd = defaultdict(list)
    for expt in expts:
        for k, v in expt.__dict__.items():
            dd[k].append(v)
    return dd
