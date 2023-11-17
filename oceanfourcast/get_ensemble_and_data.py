import os
from oceanfourcast import evaluation as ev, load_numpy as load
import numpy as np
import torch
import time
import glob


def get_ensemble_and_data(expt_dir, want_init_time, timesteps=200):

    ensembles_dir = './ensembles2'
    data_file = './data/ofn_run3_2_data/wind/run3_2/dynDiags2.npy'
    global_stats_file = './data/ofn_run3_2_data/wind/run3_2/dynDiagsGlobalStats2D.npz'
    expt1 = ev.Experiment(expt_dir)

    means = np.load(global_stats_file)['timemeans']
    means = np.concatenate((means, np.zeros((1, 248, 248))), axis=0)
    stdevs = np.load(global_stats_file)['timestdevs']
    stdevs = np.concatenate((stdevs, np.ones((1, 248, 248))), axis=0)

    files = glob.glob(os.path.join(expt_dir, ensembles_dir, '*.npy'))

    data = np.load(data_file, mmap_mode='r')
    spinup = expt1.spinupts
    rmses = []
    accs = []
    rmses_clim = []
    rmses_persistence = []
    for fstr in files:
        init_time = int(fstr.split('/')[-1].split('e')[-1].split('.')[0])
        if want_init_time == init_time:
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
            return pred, datanow, predanom, datandim
