import os
from oceanfourcast import evaluation as ev, load_numpy as load
import numpy as np
import torch
import click
import time
import glob


@click.command()
@click.option("--expt_dir", default="./")
@click.option("--ensembles_dir", default='./ensembles2')
@click.option("--data_file",
              default='../data/ofn_run3_2_data/wind/run3_2/dynDiags2.npy')
@click.option(
    "--global_stats_file",
    default="../data/ofn_run3_2_data/wind/run3_2/dynDiagsGlobalStats2D.npz")
@click.option("--timesteps", default=200)
def main(expt_dir, ensembles_dir, data_file, global_stats_file, timesteps):

    expt1 = ev.Experiment(expt_dir)

    means = np.load(global_stats_file)['timemeans']
    means = np.concatenate((means, np.zeros((1, 248, 248))), axis=0)
    stdevs = np.load(global_stats_file)['timestdevs']
    stdevs = np.concatenate((stdevs, np.ones((1, 248, 248))), axis=0)

    files = glob.glob(ensembles_dir + '/*.npy')

    data = np.load(data_file, mmap_mode='r')
    spinup = expt1.spinupts
    rmses = []
    accs = []
    rmses_clim = []
    rmses_persistence = []
    for fstr in files:
        init_time = int(fstr.split('/')[-1].split('e')[-1].split('.')[0])
        with open(fstr, 'rb') as f:
            fsz = os.fstat(f.fileno()).st_size
            predanom = np.load(f)
            while f.tell() < fsz:
                predanom = np.vstack((predanom, np.load(f)))
        pred = predanom * stdevs[:-1] + means[:-1]
        init_time = init_time + spinup
        datanow = data[init_time:(init_time + timesteps * expt1.tslag +
                                  1):expt1.tslag, :-1]
        datandim = (datanow - means[:-1]) / (stdevs[:-1] + 1e-6)

        if pred.shape[0] > datanow.shape[0]:
            pred = pred[1:]
            predanom = predanom[1:]
        rmse = np.mean((pred - datanow)**2, axis=(-2, -1))
        rmse_clim = np.mean((datanow - means[:-1])**2, axis=(-2, -1))
        rmse_persistence = np.mean((datanow - pred[0])**2, axis=(-2, -1))
        acc = np.sum(datandim * predanom, axis=(-2, -1)) / np.sqrt(
            np.sum(datandim**2, axis=(-2, -1)) *
            np.sum(predanom**2, axis=(-2, -1)))
        rmses.append(rmse)
        rmses_clim.append(rmse_clim)
        rmses_persistence.append(rmse_persistence)
        accs.append(acc)
        print(f'{fstr} done...')
    rmses = stack(rmses)
    rmses_clim = stack(rmses_clim)
    rmses_persistence = stack(rmses_persistence)
    accs = stack(accs)
    np.savez(os.path.join(expt_dir, 'ensemble_metrics_new.npz'),
             rmse=rmses,
             rmse_clim=rmses_clim,
             rmse_persistence=rmses_persistence,
             acc=accs)


def stack(list_):
    return np.vstack([a[np.newaxis] for a in list_])


if __name__ == "__main__":
    main()
