import os
import matplotlib.pyplot as plt
import json
import torch
from oceanfourcast import load, fourcastnet

class Experiment():
    def __init__(self, expt_dir, name):
        self.name = name
        self.expt_dir = expt_dir
        with open(os.path.join(expt_dir, "logfile.json"), 'r') as f:
            logs = json.load(f)
        for k, v in logs.items():
            setattr(self, k, v)

    def plot_train_loss(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1,1)
        ax.plot(self.training_loss, label=self.name)
        ax.set_xlabel('Minibatch')
        ax.set_ylabel('Train Loss')
        return ax.get_figure()

    def plot_train_valid_loss(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1,1)
        ax.plot(self.avg_training_loss, label=self.name + ' train loss')
        ax.plot(self.validation_loss, label=self.name + ' valid loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.grid()
        return ax.get_figure()

    def recreate_model(self, epoch=None, device='cpu'):
        self.model = fourcastnet.AFNONet(embed_dim=self.embed_dims,
                                         patch_size=self.patch_size,
                                         sparsity=self.sparsity,
                                         img_size=[self.image_height, self.image_width],
                                         in_channels=self.in_channels,
                                         out_channels=self.out_channels,
                                         affine_batchnorm=self.affine_batchnorm,
                                         drop_rate=self.drop_rate)
        if epoch is None:
            epoch = self.best_vloss_epoch
        model_path = os.path.join(self.expt_dir, f'model_epoch_{epoch}')
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

    def truth_compare_one_timestep(self, data_file=None, timestep=30, vmax=None, vmin=None, labels=None, cmaps=None):
        model = self.model
        if data_file is None:
            data_file = self.data_file
        ds = load.OceanDataset(data_file, for_validate=True, spinupts=self.spinupts, tslag=self.tslag)

        yi, yip1 = torch.tensor(ds[timestep])       # yi, yi + tau
        yi = yi.unsqueeze(0)
        yip1 = yip1.unsqueeze(0)
        yip1hat = model(yi)                         # fourcastnet predicted yi + tau

        stdev = torch.unsqueeze(torch.unsqueeze(torch.sqrt(model.batch_norm.running_var), -1), -1) + model.batch_norm.eps
        mean = torch.unsqueeze(torch.unsqueeze(model.batch_norm.running_mean, -1), -1)
        if model.batch_norm.weight is not None:
            wt = torch.unsqueeze(torch.unsqueeze(model.batch_norm.weight, -1), -1)
        else:
            wt = 1.0
        if model.batch_norm.bias is not None:
            bias = torch.unsqueeze(torch.unsqueeze(model.batch_norm.bias, -1), -1)
        else:
            bias = 0.0
        yip1hat = (yip1hat-bias)*stdev/wt + mean

        if cmaps is None:
            cmaps = ['RdBu_r','RdBu_r','RdBu_r','RdBu_r','RdYlBu_r','RdYlBu_r','RdYlBu_r','RdYlBu_r','RdYlBu_r']
        if labels is None:
            labels = ['U_surf', 'umid', 'V_surf', 'vmid', 'T_surf', 'thetamid', 'P_surf', 'pmid', 'pbot']
        if vmax is None:
            vmax = [0.6,   0.2,  1.25,  0.3, 28, 7,   7, 22, 32]
        if vmin is None:
            vmin = [-0.6, -0.2, -1.25, -0.3,  0, 0, -10,  8, 13]
        with plt.style.context(('labelsize15')):
            fig, ax = plt.subplots(9,3, sharex=True, sharey=True, figsize=(15,35))
            lon = ds.ds.X
            lat = ds.ds.Y
            for i in range(self.out_channels):
                im = ax[i,0].pcolormesh(lon, lat, yi.squeeze()[i]                      , vmin=vmin[i], vmax=vmax[i], cmap=cmaps[i])
                fig.colorbar(im, ax=ax[i,0])
                im = ax[i,1].pcolormesh(lon, lat, yip1.squeeze()[i]                    , vmin=vmin[i], vmax=vmax[i], cmap=cmaps[i])
                fig.colorbar(im, ax=ax[i,1])
                im = ax[i,2].pcolormesh(lon, lat, yip1hat.detach().numpy().squeeze()[i], vmin=vmin[i], vmax=vmax[i], cmap=cmaps[i])
                fig.colorbar(im, ax=ax[i,2])
                ax[i, 0].set_title(f'{labels[i]}, Initial ($t=0$)')
                ax[i, 1].set_title(f'{labels[i]}, Truth ($t=\Delta T$)')
                ax[i, 2].set_title(f'{labels[i]}, FourCastNet ($t=\Delta T$)')


            for axc in ax[-1,:]:
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
