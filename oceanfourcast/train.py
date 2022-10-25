import time
import os
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import BatchSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
from functools import partial
from oceanfourcast import load, fourcastnet
import importlib
importlib.reload(load)
importlib.reload(fourcastnet)


def train_one_epoch(epoch, model, criterion, data_loader, optimizer, summarylogger, device):
    running_loss = 0.
    last_loss =  0.
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
            print(f'batch {i+1}, loss: {last_loss}')
            tb_x = epoch * len(data_loader) + i + 1
            summarylogger.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return avg_loss/(i+1)

def main(data_location=None, epochs=5, batch_size=5, lr=5e-4, embed_dims=256, patch_size=8,
         sparsity=1e-2, device='cpu', tslag=3, spinupts=0, normalize=False, drop_rate=0.5,
         in_channels=9, out_channels=9):

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

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summarylogger = SummaryWriter(f'ofn_trainer_{timestamp}')

    best_vloss = 1000000.
    best_vloss_epoch = 1
    for epoch in range(1, epochs+1):
        print(f'EPOCH {epoch}:')

        model.train(True)
        avg_loss = train_one_epoch(epoch, model, criterion, train_dataloader, optimizer, summarylogger, device)
        model.train(False)

        running_vloss = 0.0
        with torch.no_grad():
            for i, (x,y) in enumerate(validation_dataloader):
                x = x.to(device)
                y = y.to(device)

                out = model(x)
                #y = model.batch_norm(y)

                vloss = criterion(out, y)
                running_vloss += vloss
            avg_vloss = running_vloss / (i+1)

            # Log the running loss averaged per batch
            # for both training and validation
            summarylogger.add_scalars('Training vs. Validation Loss',
                                    { 'Training' : avg_loss, 'Validation' : avg_vloss }, epoch)

            print(f'LOSS train: {avg_loss}, valid: {avg_vloss}')

            summarylogger.flush()

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                best_vloss_epoch = epoch
                model_path = f'model_{timestamp}_{epoch}'
                torch.save(model.state_dict(), model_path)

    summarylogger.add_hparams({'lr': lr, 'epochs': epochs, 'batch_size': batch_size,
                               'embed_dims': embed_dims, 'patch_size': patch_size,
                               'sparsity': sparsity, 'data_location': data_location},
                              {'hparam/best_vloss': best_vloss, 'hparam/best_vloss_epoch': best_vloss_epoch})
    summarylogger.flush()
    summarylogger.close()
    train_dataset.close()
    validation_dataset.close()
