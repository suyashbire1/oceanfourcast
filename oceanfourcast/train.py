import time
import os
import json
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
         max_runtime_hours=11.5):


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

    training_loss_logger = []
    avg_training_loss_logger = []
    validation_loss_logger = []

    best_vloss = 1000000.
    best_vloss_epoch = 1
    for epoch in range(1, epochs+1):
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
            print('Stopping due to wallclock limit...')
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
