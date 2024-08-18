import os
from oceanfourcast import evaluation as ev, load_numpy as load
import numpy as np
import torch
import click
import time


@click.command()
@click.option("--expt_dir", default="./")
@click.option("--n_ensembles", default=15)
@click.option("--out_dir", default='./ensembles_lowres')
@click.option("--data_file",
              default='./data/ofn_run3_2_data/wind/run3_2/dynDiags2.npy')
@click.option("--timesteps", default=200)
@click.option("--save_file", default=True)
@click.option("--skip", default=2)
def main(expt_dir, n_ensembles, out_dir, data_file, timesteps, save_file,
         skip):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(os.path.join(expt_dir, out_dir), exist_ok=True)

    #ni = np.random.choice(2345, n_ensembles)
    ni = np.array([
        387, 1860, 701, 1056, 1358, 1159, 49, 463, 676, 833, 1531, 189, 1972,
        2030, 35
    ])
    ni = ni[:n_ensembles]

    for i, init_time in enumerate(ni):
        if save_file:
            save_file = os.path.join(expt_dir, out_dir, f'e{init_time}.npy')
        else:
            save_file = None
        with torch.no_grad():
            expt1 = ev.Experiment(expt_dir)
            st = time.time()
            expt1.recreate_model(mlp=True, device=device, skip=skip)
            expt1.create_forward_scenario_lowres(init_time,
                                                 timesteps,
                                                 data_file=data_file,
                                                 save_file=save_file,
                                                 device=device,
                                                 skip=skip)
            et = time.time()
        print(
            f'\nTime:{et-st}s, init_time={init_time} done ({i/n_ensembles*100:0.2f}%).'
        )


if __name__ == "__main__":
    main()
