import os
from oceanfourcast import evaluation as ev, load_numpy as load
import numpy as np
import torch
import click
import time
import glob
import multiprocessing

import xarray as xr
import matplotlib.pyplot as plt
# import ffmpeg
import numpy as np
import subprocess
from joblib import Parallel, delayed

plt.rcParams.update({"text.usetex": False, "font.family": "Serif"})

cpu_count = int(multiprocessing.cpu_count() / 2)

plt.ioff()


@click.command()
@click.option("--expt_dir", default="./")
@click.option("--init_time")
@click.option("--ensembles_dir", default='./ensembles2')
@click.option("--data_file",
              default='../data/ofn_run3_2_data/wind/run3_2/dynDiags2.npy')
@click.option(
    "--global_stats_file",
    default="../data/ofn_run3_2_data/wind/run3_2/dynDiagsGlobalStats2D.npz")
@click.option("--timesteps", default=200)
@click.option("--channel", default=9)
@click.option("--vmax", default=5)
@click.option("--vmin", default=-5)
@click.option("--divisor", default=1)
@click.option("--dpi", default=150)
@click.option("--frame_rate", default=10)
def main(expt_dir, init_time, ensembles_dir, data_file, global_stats_file,
         timesteps, channel, vmax, vmin, divisor, dpi, frame_rate):

    channel_to_var = [
        r'U (m$\,$s$^{-1}$)', r'U (m$\,$s$^{-1}$)', r'V (m$\,$s$^{-1}$)',
        r'V (m$\,$s$^{-1}$)', r'SST ($^{\circ}C$)', r'T ($^{\circ}C$)',
        r'$P/\rho_0$ (m$\,$s$^{-2}$)', r'$P/\rho_0$ (m$\,$s$^{-2}$)',
        r'$P/\rho_0$ (m$\,$s$^{-2}$)', r'$\Psi$ (Sv)'
    ]
    expt1 = ev.Experiment(expt_dir)

    h, w = expt1.image_height, expt1.image_width
    lon = np.linspace(0, 62, w)
    lat = np.linspace(10, 72, h)

    means = np.load(global_stats_file)['timemeans']
    means = np.concatenate((means, np.zeros((1, h, w))), axis=0)
    stdevs = np.load(global_stats_file)['timestdevs']
    stdevs = np.concatenate((stdevs, np.ones((1, h, w))), axis=0)

    files = glob.glob(ensembles_dir + '/*.npy')

    data = np.load(data_file, mmap_mode='r')
    spinup = expt1.spinupts
    modelstrs = {'fourcastnet': 'AFNO', 'fno': 'FNO', 'unet': 'U-Net'}
    modelstr = modelstrs[expt1.modelstr]
    for fstr in files:
        found_init_time = int(fstr.split('/')[-1].split('e')[-1].split('.')[0])
        if found_init_time == int(init_time):
            print('Reading data...')
            with open(fstr, 'rb') as f:
                fsz = os.fstat(f.fileno()).st_size
                predanom = np.load(f)
                while f.tell() < fsz:
                    predanom = np.vstack((predanom, np.load(f)))
            pred = predanom * stdevs[:-1] + means[:-1]
            found_init_time = found_init_time + spinup
            datanow = data[found_init_time:(found_init_time +
                                            timesteps * expt1.tslag +
                                            1):expt1.tslag, :-1]
            print('Reading data done...')

            def make_figure(i):
                # for i in range(datanow.shape[0]):
                fig, ax = plt.subplots(1,
                                       2,
                                       sharex=True,
                                       sharey=True,
                                       figsize=(10, 5))
                print(f"Plotting channel {channel} composite plot: {i}...")
                im = ax[0].pcolormesh(lon,
                                      lat,
                                      datanow[i].squeeze()[channel] / divisor,
                                      cmap='RdBu_r',
                                      vmin=vmin,
                                      vmax=vmax)
                im = ax[1].pcolormesh(lon,
                                      lat,
                                      pred[i].squeeze()[channel] / divisor,
                                      cmap='RdBu_r',
                                      vmin=vmin,
                                      vmax=vmax)

                # ax[0].set_title(
                #     f'{channel_to_var[channel]}, Truth (t = {i*expt1.tslag} days)'
                # )
                # ax[1].set_title(
                #     f'{channel_to_var[channel]}, {modelstr} (t = {i*expt1.tslag} days)'
                # )
                ax[0].set_title(f'Truth (t = {i*expt1.tslag} days)')
                ax[1].set_title(f'{modelstr} (t = {i*expt1.tslag} days)')

                for axc in ax.ravel():
                    axc.set_xlabel(r'Lon ($^{\circ}$)')
                    axc.set_aspect(1)
                ax[0].set_ylabel(r'Lat ($^{\circ}$)')

                cb = fig.colorbar(im, ax=ax.ravel(), shrink=0.3)
                cb.ax.set_ylabel(channel_to_var[channel])
                fig.savefig(f'temp_figs_{i:05d}.png',
                            bbox_inches='tight',
                            dpi=dpi)
                plt.close(fig)

            Parallel(n_jobs=cpu_count)(delayed(make_figure)(i)
                                       for i in range(datanow.shape[0]))

            break

    subprocess.run([
        "ffmpeg", "-r", f"{frame_rate}", "-pattern_type", "glob", "-i",
        "*.png", f"{ensembles_dir}/e{init_time}_psi.mp4"
    ],
                   check=True)
    png_files = [
        file for file in os.listdir(os.getcwd()) if file.endswith(".png")
    ]
    print(f"Deleting {len(png_files)} leftover png files...")
    [os.remove(file) for file in png_files]


if __name__ == "__main__":
    main()
