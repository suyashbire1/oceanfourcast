import os
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


def train_one_epoch(epoch, model, criterion, data_loader, optimizer, summarylogger):
    running_loss = 0.
    last_loss =  0.
    for i, batch in enumerate(data_loader):

        # Load data
        x, y = batch[0][:-1], batch[0][1:]
        #x = x.unsqueeze(0)
        #y = y.unsqueeze(0)

        # Zero gradients for every batch
        optimizer.zero_grad()

        # Make predictions for this batch
        out = model(x)

        # Compute the loss and its gradients
        loss = criterion(out, y)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 10 == 9:
            last_loss = running_loss / 10 # loss per batch
            print(f'batch {i+1}, loss: {last_loss}')
            tb_x = epoch * len(data_loader) + i + 1
            summarylogger.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

def main():
    epochs = 3
    batch_size = 5
    lr = 5e-4
    embed_dims = 256
    patch_size = 8
    sparsity = 0

    # input size
    h, w = 248, 248
    x_c, y_c = 6, 6

    # fix the seed for reproducibility
    seed = 1024
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_dataset = load.OceanDataset("/home/suyash/Documents/data/")
    #train_datasampler = BatchSampler(train_dataset, batch_size= batch_size=batch_size, drop_last=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)#, batch_sampler=train_datasampler)

    validation_dataset = load.OceanDataset("/home/suyash/Documents/data/", for_validate=True)
    # validation_datasampler = BatchSampler(validation_dataset, batch_size=2, drop_last=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)#, batch_sampler=validation_datasampler)

    model = fourcastnet.AFNONet(embed_dims, patch_size, sparsity, img_size=[h, w], in_chans=x_c, out_chans=y_c, norm_layer=partial(nn.LayerNorm, eps=1e-6))

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=lr, betas=(0.9, 0.95))

    # optimizer = torch.optim.AdamW(lr=lr, betas=(0.9, 0.95))
    # loss_scaler = torch.cpu.amp.GradScaler(enabled=True)
    # criterion = nn.MSELoss()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summarylogger = SummaryWriter(f'ofn_trainer_{timestamp}')

    for epoch in range(epochs):
        print('EPOCH {}:'.format(epoch + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch, model, criterion, train_dataloader, optimizer, summarylogger)

        # We don't need gradients on to do reporting
        model.train(False)

        running_vloss = 0.0
        for i, vdata in enumerate(validation_dataloader):
            # Load data
            x, y = vdata[0][:-1], vdata[0][1:]
            #x = x.unsqueeze(0)
            #y = y.unsqueeze(0)

            # Make predictions for this batch
            out = model(x)

            # Compute the loss and its gradients
            vloss = criterion(out, y)
            running_vloss += vloss
        avg_vloss = running_vloss / (i+1)

        print(f'LOSS train: {avg_loss}, valid: {avg_vloss}')

        # Log the running loss averaged per batch
        # for both training and validation
        summarylogger.add_scalars('Training vs. Validation Loss',
                                { 'Training' : avg_loss, 'Validation' : avg_vloss }, epoch + 1)
        summarylogger.flush()

        # # Track best performance, and save the model's state
        # if avg_vloss < best_vloss:
        #     best_vloss = avg_vloss
        #     model_path = 'model_{}_{}'.format(timestamp, epoch)
        #     torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    pass
    # ngpus = torch.cuda.device_count()
    # torch.multiprocessing.spawn(main, args=(), nprocs=ngpus)
