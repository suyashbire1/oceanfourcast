import time
import os
import subprocess
import json
import glob
import argparse
import click
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, random_split, ConcatDataset, Subset
from functools import partial
from oceanfourcast import load_numpy as load
from oceanfourcast import fourcastnet
import importlib

importlib.reload(load)
importlib.reload(fourcastnet)


@click.command()
@click.option("--name", default="experiment")
@click.option("--output_dir", default="./")
@click.option("--data_file", default=None)
@click.option("--epochs", default=5)
@click.option("--batch_size", default=5)
@click.option("--learning_rate", default=5e-4)
@click.option("--embed_dims", default=256)
@click.option("--patch_size", default=8)
@click.option("--depth", default=12)
@click.option("--num_blocks", default=8)
@click.option("--mlp_ratio", default=4)
@click.option("--sparsity", default=1e-2)
@click.option("--device", default='cpu')
@click.option("--tslag", default=3)
@click.option("--spinupts", default=0)
@click.option("--drop_rate", default=0.5)
@click.option("--out_channels", default=9)
@click.option("--max_runtime_hours", default=11.5)
@click.option("--resume_from_chkpt", default=False)
@click.option("--optimizerstr", default='adam')
@click.option("--modelstr", default="fourcastnet")
@click.option("--fine_tune", default=False)
@click.option("--mmap_mode", default=None)
@click.option("--nfmodes", default=16)
@click.option("--fnonorm", default=None)
@click.option("--kecons", default=False)
def main(name, output_dir, data_file, epochs, batch_size, learning_rate,
         embed_dims, patch_size, depth, num_blocks, mlp_ratio, sparsity,
         device, tslag, spinupts, drop_rate, out_channels, max_runtime_hours,
         resume_from_chkpt, optimizerstr, modelstr, fine_tune, mmap_mode,
         nfmodes, fnonorm, kecons):

    start_time = datetime.now()
    end_time = start_time + timedelta(hours=max_runtime_hours)

    print(f'Run started on ', start_time.strftime('%Y%m%d_%H%M%S'))

    os.makedirs(output_dir, exist_ok=True)

    # fix the seed for reproducibility
    seed = 1024
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    if data_file is not None:
        # train_dataset = load.OceanDataset(data_file, spinupts=spinupts, tslag=tslag)
        global_dataset = load.OceanDataset(data_file,
                                           spinupts=spinupts,
                                           tslag=tslag,
                                           device=device,
                                           fine_tune=fine_tune)
        h, w = global_dataset.img_size
        # in_channels = len(train_dataset.channels)
        in_channels = global_dataset.channels

        b = len(global_dataset)
        train_set_len = int(0.9 * b)
        valid_set_len = b - train_set_len
        train_dataset, validation_dataset = random_split(
            global_dataset, [train_set_len, valid_set_len])
    else:
        print('Loading datasets...')
        path_ = "/home/bire/nobackup/"
        dataset1 = load.OceanDataset(
            path_ + "ofn_run3_2_data/wind/run3_2/dynDiags2.npy",
            spinupts=spinupts,
            tslag=tslag,
            device=device,
            fine_tune=fine_tune,
            multi_expt_normalize=True,
            mmap_mode=mmap_mode)
        h, w = dataset1.img_size
        in_channels = dataset1.channels
        means = dataset1.means
        stdevs = dataset1.stdevs
        dataset2 = load.OceanDataset(
            path_ + "ofn_run3_2_data/wind/run3_2_less_wind/dynDiags2.npy",
            spinupts=spinupts,
            tslag=tslag,
            device=device,
            fine_tune=fine_tune,
            multi_expt_normalize=True,
            mmap_mode=mmap_mode)
        # dataset3 = load.OceanDataset(
        #     path_+"ofn_run3_2_data/run3_2_less_flux/dynDiags.npy",
        #     spinupts=spinupts,
        #     tslag=tslag,
        #     device=device,
        #     fine_tune=fine_tune,
        #     multi_expt_normalize=True, mmap_mode=mmap_mode)
        dataset4 = load.OceanDataset(
            path_ + "ofn_run3_2_data/wind/run3_2_more_wind/dynDiags2.npy",
            spinupts=spinupts,
            tslag=tslag,
            device=device,
            fine_tune=fine_tune,
            multi_expt_normalize=True,
            mmap_mode=mmap_mode)
        # dataset5 = load.OceanDataset(
        #     path_+"ofn_run3_2_data/run3_2_more_flux/dynDiags.npy",
        #     spinupts=spinupts,
        #     tslag=tslag,
        #     device=device,
        #     fine_tune=fine_tune,
        #     multi_expt_normalize=True, mmap_mode=mmap_mode)
        ds1_len = len(dataset1)
        validation_dataset = Subset(dataset1, range(ds1_len // 2))
        train_dataset = ConcatDataset(
            (dataset2, dataset4, Subset(dataset1, range(ds1_len // 2,
                                                        ds1_len))))
        #train_dataset = Subset(dataset1, range(ds1_len // 5, ds1_len))
        print('Done loading datasets...')

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  drop_last=True,
                                  shuffle=True)

    # validation_dataset = load.OceanDataset(data_file, for_validate=True, spinupts=spinupts, tslag=tslag)
    validation_dataloader = DataLoader(validation_dataset,
                                       batch_size=batch_size,
                                       drop_last=True)

    if modelstr == 'fourcastnet':
        model = fourcastnet.AFNONet(embed_dim=embed_dims,
                                    patch_size=patch_size,
                                    sparsity=sparsity,
                                    img_size=[h, w],
                                    in_channels=in_channels,
                                    out_channels=out_channels,
                                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                    device=device,
                                    drop_rate=drop_rate,
                                    mlp_ratio=mlp_ratio,
                                    depth=depth,
                                    n_blocks=num_blocks).to(device)
    elif modelstr == 'unet':
        from oceanfourcast import unet
        model = unet.UNet(n_channels=in_channels,
                          n_classes=out_channels,
                          device=device)
    elif modelstr == 'fno':
        from neuralop.models import FNO
        pos_emb = fourcastnet.PosEmbed([h, w], device=device)
        fno_model = FNO(n_modes=(nfmodes, nfmodes),
                        n_layers=depth,
                        hidden_channels=embed_dims,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        norm=fnonorm,
                        use_mlp=True,
                        mlp=dict(dropout=drop_rate, expansion=mlp_ratio),
                        device=device).to(device)
        model = nn.Sequential(pos_emb, fno_model)
        model.Co = out_channels
    else:
        print(f'argument modelstr {modelstr} invalid')

    criterion = nn.MSELoss()
    learnable_params = list(model.parameters())
    lag_multiplier = torch.rand(2, requires_grad=True, device=device)
    learnable_params += [lag_multiplier]
    optimizers = {
        'adam':
        torch.optim.Adam(learnable_params, lr=learning_rate,
                         betas=(0.9, 0.95)),
        'adamw':
        torch.optim.AdamW(learnable_params,
                          lr=learning_rate,
                          betas=(0.9, 0.95)),
        'sgd':
        torch.optim.SGD(learnable_params, lr=learning_rate)
    }

    if resume_from_chkpt:
        pattern = os.path.join(output_dir, "chkpt_epoch_*")
        chkpt_file = get_latest_checkpoint_file(pattern)
        print(f'Resuming from checkpoint {chkpt_file}...')
        checkpoint = torch.load(chkpt_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizerstr = checkpoint['optimizerstr']
        optimizer = optimizers[optimizerstr]
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        begin_epoch = checkpoint['epoch'] + 1
        if fine_tune:
            best_vloss = 1000000.
            best_vloss_epoch = 1
        else:
            best_vloss = checkpoint['best_vloss']
            best_vloss_epoch = checkpoint['best_vloss_epoch']
        training_loss_logger = checkpoint['training_loss_logger']
        avg_training_loss_logger = checkpoint['avg_training_loss_logger']
        validation_loss_logger = checkpoint['validation_loss_logger']
    else:
        optimizer = optimizers[optimizerstr]
        begin_epoch = 1
        training_loss_logger = []
        avg_training_loss_logger = []
        validation_loss_logger = []
        best_vloss = 1000000.
        best_vloss_epoch = 1

    if fine_tune:
        if kecons:
            train_func = train_one_epoch_finetune_kecons
            validate_func = validate_one_epoch_finetune_kecons
            kwargs = dict(lag_multiplier=lag_multiplier,
                          means=means,
                          stdevs=stdevs)
        else:
            train_func = train_one_epoch_finetune
            validate_func = validate_one_epoch_finetune
            kwargs = dict()
    else:
        train_func = train_one_epoch
        validate_func = validate_one_epoch
        kwargs = dict()

    for epoch in range(begin_epoch, epochs + 1):
        epoch_start_time = datetime.now()
        print(
            f'EPOCH {epoch}:-----------------------------------------------------------'
        )

        model.train(True)
        print('Training...')
        avg_loss = train_func(epoch, model, criterion, train_dataloader,
                              optimizer, device, training_loss_logger,
                              **kwargs)
        model.train(False)
        print('Validating...')
        avg_vloss = validate_func(model, criterion, validation_dataloader,
                                  device, **kwargs)
        print(f'LOSS train: {avg_loss}, valid: {avg_vloss}')
        print(f'Epoch evaluation time: {(datetime.now()-epoch_start_time)}')
        avg_training_loss_logger.append(avg_loss)
        validation_loss_logger.append(avg_vloss)

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            best_vloss_epoch = epoch
            model_path = os.path.join(output_dir, f'model_epoch_{epoch}')
            torch.save(model.state_dict(), model_path)

            print('Saving intermediate checkpoint...')
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizerstr': optimizerstr,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_vloss': best_vloss,
                    'best_vloss_epoch': best_vloss_epoch,
                    'training_loss_logger': training_loss_logger,
                    'avg_training_loss_logger': avg_training_loss_logger,
                    'validation_loss_logger': validation_loss_logger
                }, os.path.join(output_dir, f"chkpt_epoch_{epoch}"))

        if datetime.now() > end_time:
            print('Stopping due to wallclock limit...')
            break

    print('Saving final checkpoint...')
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizerstr': optimizerstr,
            'optimizer_state_dict': optimizer.state_dict(),
            'best_vloss': best_vloss,
            'best_vloss_epoch': best_vloss_epoch,
            'training_loss_logger': training_loss_logger,
            'avg_training_loss_logger': avg_training_loss_logger,
            'validation_loss_logger': validation_loss_logger
        }, os.path.join(output_dir, f"chkpt_final_epoch_{epoch}"))

    print('Writing logs...')
    logfile_data = dict(name=name,
                        data_file=data_file,
                        epochs=epochs,
                        batch_size=batch_size,
                        nfmodes=nfmodes,
                        learning_rate=learning_rate,
                        optimizerstr=optimizerstr,
                        embed_dims=embed_dims,
                        patch_size=patch_size,
                        image_height=h,
                        image_width=w,
                        num_blocks=num_blocks,
                        mlp_ratio=mlp_ratio,
                        depth=depth,
                        sparsity=sparsity,
                        device=device,
                        tslag=tslag,
                        fnonorm=fnonorm,
                        spinupts=spinupts,
                        modelstr=modelstr,
                        drop_rate=drop_rate,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        fine_tune=fine_tune,
                        best_vloss=best_vloss,
                        best_vloss_epoch=best_vloss_epoch,
                        runtime=str(datetime.now() - start_time),
                        training_loss=training_loss_logger,
                        avg_training_loss=avg_training_loss_logger,
                        validation_loss=validation_loss_logger,
                        version=get_git_revision_hash(load))

    with open(os.path.join(output_dir, "logfile.json"), "w") as f:
        f.write(json.dumps(logfile_data, indent=4))

    # train_dataset.close()
    # validation_dataset.close()


def train_one_epoch(epoch, model, criterion, data_loader, optimizer, device,
                    training_loss_logger, **kwargs):
    running_loss = 0.
    avg_loss = 0.
    print_every = 10
    iters = len(data_loader) // print_every
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, iters)
    for i, (x, y) in enumerate(data_loader):
        x = x.to(device, dtype=torch.float)
        y = y.to(device, dtype=torch.float)

        optimizer.zero_grad()

        out = model(x)

        loss = criterion(out, y[:, :model.Co])
        loss.backward()

        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        avg_loss += loss.item()
        if i % print_every == 9:
            scheduler.step()
            last_loss = running_loss / 10  # loss per batch
            training_loss_logger.append(last_loss)
            print(
                f'batch {i+1}, loss: {last_loss:0.2E}, lr: {scheduler.get_last_lr()[0]:0.2E}'
            )
            running_loss = 0.

    return avg_loss / (i + 1)


def train_one_epoch_finetune(epoch, model, criterion, data_loader, optimizer,
                             device, training_loss_logger, **kwargs):
    running_loss = 0.
    avg_loss = 0.
    print_every = 10
    iters = len(data_loader) // print_every
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, iters)
    for i, (x, (y1, y2)) in enumerate(data_loader):
        x = x.to(device, dtype=torch.float)
        y1 = y1.to(device, dtype=torch.float)
        y2 = y2.to(device, dtype=torch.float)

        optimizer.zero_grad()

        out1 = model(x)
        loss1 = criterion(out1, y1[:, :model.Co])
        out1 = torch.cat((out1, x[:, model.Co:]), dim=1)
        out2 = model(out1)

        loss = loss1 + criterion(out2, y2[:, :model.Co])
        loss.backward()

        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        avg_loss += loss.item()
        if i % print_every == 9:
            scheduler.step()
            last_loss = running_loss / 10  # loss per batch
            training_loss_logger.append(last_loss)
            print(
                f'batch {i+1}, loss: {last_loss:0.2E}, lr: {scheduler.get_last_lr()[0]:0.2E}'
            )
            running_loss = 0.

    return avg_loss / (i + 1)


def train_one_epoch_finetune_kecons(epoch, model, criterion, data_loader,
                                    optimizer, device, training_loss_logger,
                                    **kwargs):
    running_loss = 0.
    avg_loss = 0.
    print_every = 10
    iters = len(data_loader) // print_every
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, iters)
    lm = kwargs.get('lag_multiplier')
    means = torch.tensor(kwargs.get('means'), device=device, dtype=torch.float)
    stdevs = torch.tensor(kwargs.get('stdevs'),
                          device=device,
                          dtype=torch.float)
    for i, (x, (y1, y2)) in enumerate(data_loader):
        x = x.to(device, dtype=torch.float)
        y1 = y1.to(device, dtype=torch.float)
        y2 = y2.to(device, dtype=torch.float)

        optimizer.zero_grad()

        out1 = model(x)
        loss1 = criterion(out1, y1[:, :model.Co])
        out1 = torch.cat((out1, y1[:, model.Co:]), dim=1)
        out2 = model(out1)
        loss2 = criterion(out2, y2[:, :model.Co])

        usurf = x[:, 0] * stdevs[0] + means[0]
        ubot = x[:, 1] * stdevs[1] + means[1]
        vsurf = x[:, 2] * stdevs[2] + means[2]
        vbot = x[:, 3] * stdevs[3] + means[3]
        ke0 = torch.mean((usurf + ubot)**2 + (vsurf + vbot)**2)

        usurf = out1[:, 0] * stdevs[0] + means[0]
        ubot = out1[:, 1] * stdevs[1] + means[1]
        vsurf = out1[:, 2] * stdevs[2] + means[2]
        vbot = out1[:, 3] * stdevs[3] + means[3]
        ke1 = torch.mean((usurf + ubot)**2 + (vsurf + vbot)**2)

        usurf = out2[:, 0] * stdevs[0] + means[0]
        ubot = out2[:, 1] * stdevs[1] + means[1]
        vsurf = out2[:, 2] * stdevs[2] + means[2]
        vbot = out2[:, 3] * stdevs[3] + means[3]
        ke2 = torch.mean((usurf + ubot)**2 + (vsurf + vbot)**2)

        keloss = torch.abs((ke1 - ke0) * lm[0]) + torch.abs(
            (ke2 - ke1) * lm[1])
        loss = loss1 + loss2 + keloss
        loss.backward()

        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        avg_loss += loss.item()
        if i % print_every == 9:
            scheduler.step()
            last_loss = running_loss / 10  # loss per batch
            training_loss_logger.append(last_loss)
            print(
                f'batch {i+1}, loss: {last_loss:0.2E}, lr: {scheduler.get_last_lr()[0]:0.2E}'
            )
            running_loss = 0.

    return avg_loss / (i + 1)


def validate_one_epoch(model, criterion, data_loader, device):
    with torch.no_grad():
        running_vloss = 0.0
        for i, (x, y) in enumerate(data_loader):
            x = x.to(device, dtype=torch.float)
            y = y.to(device, dtype=torch.float)

            out = model(x)

            vloss = criterion(out, y[:, :model.Co])
            running_vloss += vloss.item()
    return running_vloss / (i + 1)


def validate_one_epoch_finetune(model, criterion, data_loader, device):
    with torch.no_grad():
        running_vloss = 0.0
        for i, (x, (y1, y2)) in enumerate(data_loader):
            x = x.to(device, dtype=torch.float)
            y1 = y1.to(device, dtype=torch.float)
            y2 = y2.to(device, dtype=torch.float)

            out1 = model(x)
            loss1 = criterion(out1, y1[:, :model.Co])
            out1 = torch.cat((out1, x[:, model.Co:]), dim=1)
            out2 = model(out1)
            vloss = loss1 + criterion(out2, y2[:, :model.Co])
            running_vloss += vloss.item()
    return running_vloss / (i + 1)


def validate_one_epoch_finetune_kecons(model, criterion, data_loader, device,
                                       **kwargs):
    with torch.no_grad():
        lm = kwargs.get('lag_multiplier')
        means = torch.tensor(kwargs.get('means'),
                             device=device,
                             dtype=torch.float)
        stdevs = torch.tensor(kwargs.get('stdevs'),
                              device=device,
                              dtype=torch.float)
        running_vloss = 0.0
        for i, (x, (y1, y2)) in enumerate(data_loader):
            x = x.to(device, dtype=torch.float)
            y1 = y1.to(device, dtype=torch.float)
            y2 = y2.to(device, dtype=torch.float)

            out1 = model(x)
            loss1 = criterion(out1, y1[:, :model.Co])
            out1 = torch.cat((out1, y1[:, model.Co:]), dim=1)
            out2 = model(out1)
            loss2 = criterion(out2, y2[:, :model.Co])

            usurf = x[:, 0] * stdevs[0] + means[0]
            ubot = x[:, 1] * stdevs[1] + means[1]
            vsurf = x[:, 2] * stdevs[2] + means[2]
            vbot = x[:, 3] * stdevs[3] + means[3]
            ke0 = torch.mean((usurf + ubot)**2 + (vsurf + vbot)**2)

            usurf = out1[:, 0] * stdevs[0] + means[0]
            ubot = out1[:, 1] * stdevs[1] + means[1]
            vsurf = out1[:, 2] * stdevs[2] + means[2]
            vbot = out1[:, 3] * stdevs[3] + means[3]
            ke1 = torch.mean((usurf + ubot)**2 + (vsurf + vbot)**2)

            usurf = out2[:, 0] * stdevs[0] + means[0]
            ubot = out2[:, 1] * stdevs[1] + means[1]
            vsurf = out2[:, 2] * stdevs[2] + means[2]
            vbot = out2[:, 3] * stdevs[3] + means[3]
            ke2 = torch.mean((usurf + ubot)**2 + (vsurf + vbot)**2)
            keloss = torch.abs((ke1 - ke0) * lm[0]) + torch.abs(
                (ke1 - ke2) * lm[1])
            vloss = loss1 + loss2 + keloss
            running_vloss += vloss.item()
    return running_vloss / (i + 1)


#def check_epoch_metrics(model, dataset, device):
#    with torch.no_grad():
#    acc


def get_latest_checkpoint_file(pattern):

    # get list of files that matches pattern
    files = list(filter(os.path.isfile, glob.glob(pattern)))
    # sort by modified time
    files.sort(key=lambda x: os.path.getmtime(x))
    # get last item in list
    lastfile = files[-1]
    return lastfile


def get_git_revision_hash(module):
    cwd = os.getcwd()
    os.chdir(os.path.dirname(module.__file__))
    githash = subprocess.check_output(['git', 'rev-parse',
                                       'HEAD']).decode('ascii').strip()
    os.chdir(cwd)
    return githash


# def latitude_weighting(lat_array):
#     nlat = len(lat_array)
#     return nlat*np.cos(lat_array)/np.sum(np.cos(lat_array))

# def anomaly_correlation_coefficient_lw(xtrue, xpred, channel, lat_array):
#     L = latitude_weighting(lat_array)
#     xpredanom = (xpred[channel]-mean[channel])
#     xtrueanom = (xtrue[channel]-mean[channel])
#     numerator = np.sum(L*xpredanom*xtrueanom)
#     denominator = np.sqrt(np.sum(L*xpredanom**2)*np.sum(L*xtrueanom**2))
#     return numerator/denominator

# def relative_quantile_error_lw():
#     pass

# def rmse_lw(xtrue, xpred, channel, lat_array):
#     nx, ny = xtrue.shape[-2:]
#     L = latitude_weighting(lat_array)
#     return np.sqrt(np.sum(L*(xpred-xtrue)))/nx/ny

if __name__ == "__main__":
    main()
