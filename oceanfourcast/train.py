import time
import os
import json
import glob
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from functools import partial
from oceanfourcast import load, fourcastnet
import importlib
importlib.reload(load)
importlib.reload(fourcastnet)

def main(output_dir="./",
         data_location=None,
         epochs=5,
         batch_size=5,
         learning_rate=5e-4,
         embed_dims=256,
         patch_size=8,
         sparsity=1e-2,
         device='cpu',
         tslag=3,
         spinupts=0,
         normalize=False,
         drop_rate=0.5,
         in_channels=9,
         out_channels=9,
         max_runtime_hours=11.5,
         resume_from_chkpt=False
         ):


    start_time = datetime.now()
    end_time = start_time + timedelta(hours=max_runtime_hours)

    print(start_time.strftime('%Y%m%d_%H%M%S'))

    # fix the seed for reproducibility
    seed = 1024
    torch.manual_seed(seed)
    np.random.seed(seed)

    if data_location is None:
        data_location = "/home/suyash/Documents/data/"

    train_dataset = load.OceanDataset(data_location, spinupts=spinupts, tslag=tslag, normalize=normalize)
    h, w = train_dataset.img_size
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)

    validation_dataset = load.OceanDataset(data_location, for_validate=True, spinupts=spinupts, tslag=tslag, normalize=normalize)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, drop_last=True)

    model = fourcastnet.AFNONet(embed_dim=embed_dims,
                                patch_size=patch_size,
                                sparsity=sparsity,
                                img_size=[h, w],
                                in_channels=in_channels,
                                out_channels=out_channels,
                                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                device=device,
                                drop_rate=drop_rate).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.95))

    if resume_from_chkpt:
        pattern = os.path.join(output_dir,"chkpt_epoch_*")
        print(f'Reading checkpoint {pattern}...')
        checkpoint = torch.load(get_latest_checkpoint_file(pattern))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        begin_epoch = checkpoint['epoch'] + 1
        best_vloss = checkpoint['best_vloss']
        best_vloss_epoch = checkpoint['best_vloss_epoch']
        training_loss_logger = checkpoint['training_loss_logger']
        avg_training_loss_logger = checkpoint['avg_training_loss_logger']
        validation_loss_logger = checkpoint['validation_loss_logger']
    else:
        begin_epoch = 1
        training_loss_logger = []
        avg_training_loss_logger = []
        validation_loss_logger = []
        best_vloss = 1000000.
        best_vloss_epoch = 1

    for epoch in range(begin_epoch, epochs+1):
        print(f'EPOCH {epoch}:----------------------------------------')

        model.train(True)
        avg_loss = train_one_epoch(model, criterion, train_dataloader, optimizer, device, training_loss_logger)
        model.train(False)
        avg_vloss = validate_one_epoch(model, criterion, data_loader, device)
        print(f'LOSS train: {avg_loss}, valid: {avg_vloss}')
        avg_training_loss_logger.append(avg_loss)
        validation_loss_logger.append(avg_vloss)

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            best_vloss_epoch = epoch
            model_path = f'model_{timestamp}_{epoch}'
            torch.save(model.state_dict(), model_path)

        if datetime.now() > end_time:
            print('Stopping due to wallclock limit. Saving checkpoint')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_vloss': best_vloss,
                'best_vloss_epoch': best_vloss_epoch,
                'training_loss_logger': training_loss_logger,
                'avg_training_loss_logger': avg_training_loss_logger,
                'validation_loss_logger': validation_loss_logger},
                       os.path.join(output_dir,f"chkpt_epoch_{epoch}"))
            break

    print('Writing logs...')

    logfile_data = dict(
        data_location=data_location,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        embed_dims=embed_dims,
        patch_size=patch_size,
        sparsity=sparsity,
        device=device,
        tslag=tslag,
        spinupts=spinupts,
        normalize=normalize,
        drop_rate=drop_rate,
        in_channels=in_channels,
        out_channels=out_channels,
        best_vloss=best_vloss,
        best_vloss_epoch=best_vloss_epoch,
        runtime=str(datetime.now() - start_time),
        training_loss = training_loss_logger,
        avg_training_loss = avg_training_loss_logger,
        validation_loss = validation_loss_logger
    )

    with open(os.path.join(output_dir,"logfile.json"), "w") as f:
        f.write(json.dumps(logfile_data))

    train_dataset.close()
    validation_dataset.close()

def train_one_epoch(model, criterion, data_loader, optimizer, device, training_loss_logger):
    running_loss = 0.
    avg_loss = 0.
    for i, (x,y) in enumerate(data_loader):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        out = model(x)
        #with torch.no_grad():
        #    y = model.batch_norm(y)

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
            #y = model.batch_norm(y)

            vloss = criterion(out, y)
            running_vloss += vloss
    return running_vloss / (i+1)

def get_latest_checkpoint_file(pattern):

    # get list of files that matches pattern
    files = list(filter(os.path.isfile, glob.glob(pattern)))
    # sort by modified time
    files.sort(key=lambda x: os.path.getmtime(x))
    # get last item in list
    lastfile = files[-1]
    return lastfile
