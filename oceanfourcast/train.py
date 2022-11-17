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
from torch.utils.data import DataLoader, random_split
from functools import partial
from oceanfourcast import load_numpy as load, fourcastnet
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
def main(name, output_dir, data_file, epochs, batch_size,
         learning_rate, embed_dims, patch_size, depth,
         num_blocks, mlp_ratio, sparsity, device, tslag,
         spinupts, drop_rate, out_channels, max_runtime_hours,
         resume_from_chkpt, optimizerstr):

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

    assert data_file is not None

    # train_dataset = load.OceanDataset(data_file, spinupts=spinupts, tslag=tslag)
    global_dataset = load.OceanDataset(data_file, spinupts=spinupts, tslag=tslag, device=device)
    h, w = global_dataset.img_size
    # in_channels = len(train_dataset.channels)
    in_channels = global_dataset.channels

    b = len(global_dataset)
    train_set_len = int(0.9*b)
    valid_set_len = b - train_set_len
    train_dataset, validation_dataset = random_split(global_dataset, [train_set_len, valid_set_len])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)

    # validation_dataset = load.OceanDataset(data_file, for_validate=True, spinupts=spinupts, tslag=tslag)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, drop_last=True)

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
                                num_blocks=num_blocks).to(device)

    criterion = nn.MSELoss()
    optimizers = {
            'adam': torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.95)),
            'adamw': torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95)),
            'sgd': torch.optim.SGD(model.parameters(), lr=learning_rate)
            }

    if resume_from_chkpt:
        pattern = os.path.join(output_dir,"chkpt_epoch_*")
        chkpt_file = get_latest_checkpoint_file(pattern)
        print(f'Resuming from checkpoint {chkpt_file}...')
        checkpoint = torch.load(chkpt_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizerstr = checkpoint['optimizerstr']
        optimizer = optimizers[optimizerstr]
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        begin_epoch = checkpoint['epoch'] + 1
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

    for epoch in range(begin_epoch, epochs+1):
        epoch_start_time = datetime.now()
        print(f'EPOCH {epoch}:-----------------------------------------------------------')

        model.train(True)
        print('Training...')
        avg_loss = train_one_epoch(model, criterion, train_dataloader, optimizer, device, training_loss_logger)
        model.train(False)
        print('Validating...')
        avg_vloss = validate_one_epoch(model, criterion, validation_dataloader, device)
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

        if datetime.now() > end_time:
            print('Stopping due to wallclock limit...')
            break

    print('Saving checkpoint...')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizerstr': optimizerstr,
        'optimizer_state_dict': optimizer.state_dict(),
        'best_vloss': best_vloss,
        'best_vloss_epoch': best_vloss_epoch,
        'training_loss_logger': training_loss_logger,
        'avg_training_loss_logger': avg_training_loss_logger,
        'validation_loss_logger': validation_loss_logger},
               os.path.join(output_dir,f"chkpt_epoch_{epoch}"))

    print('Writing logs...')
    logfile_data = dict(
        name=name,
        data_file=data_file,
        epochs=epochs,
        batch_size=batch_size,
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
        spinupts=spinupts,
        drop_rate=drop_rate,
        in_channels=in_channels,
        out_channels=out_channels,
        best_vloss=best_vloss,
        best_vloss_epoch=best_vloss_epoch,
        runtime=str(datetime.now() - start_time),
        training_loss = training_loss_logger,
        avg_training_loss = avg_training_loss_logger,
        validation_loss = validation_loss_logger,
        version = get_git_revision_hash(load)
    )

    with open(os.path.join(output_dir,"logfile.json"), "w") as f:
        f.write(json.dumps(logfile_data, indent=4))

    # train_dataset.close()
    # validation_dataset.close()

def train_one_epoch(model, criterion, data_loader, optimizer, device, training_loss_logger):
    running_loss = 0.
    avg_loss = 0.
    for i, (x,y) in enumerate(data_loader):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        out = model(x)

        loss = criterion(out, y)
        loss.backward()

        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        avg_loss += loss.item()
        if i % 10 == 9:
            last_loss = running_loss / 10 # loss per batch
            training_loss_logger.append(last_loss)
            print(f'batch {i+1}, loss: {last_loss}')
            running_loss = 0.

    return avg_loss/(i+1)

def validate_one_epoch(model, criterion, data_loader, device):
    with torch.no_grad():
        running_vloss = 0.0
        for i, (x,y) in enumerate(data_loader):
            x = x.to(device)
            y = y.to(device)

            out = model(x)

            vloss = criterion(out, y)
            running_vloss += vloss.item()
    return running_vloss / (i+1)

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
    githash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    os.chdir(cwd)
    return githash

if __name__ == "__main__":
    main()
