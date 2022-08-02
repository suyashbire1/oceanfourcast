import os
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import BatchSampler
from functools import partial
from load import OceanDataset


def train_one_epoch(epoch, start_step, model, criterion, data_loder, optimizer, loss_scalar, lr_scheduler, min_loss):
    model.train()
    for step, batch in enumerate(data_loader):
        x, y = [x for x in batch[:2]]
        x = x.transpose(3,2).transpose(2,1)
        y = y.transpose(3,2).transpose(2,1)

        out = model(x)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss_scaler.scale(loss).backward()
        loss_scaler.step(optimizer)
        loss_scaler.update()



def main(local_rank):
    epochs = 80
    batch_size = 3
    lr = 5e-4

    # input size
    h, w = 720, 1440
    x_c, y_c = 20, 20

    # fix the seed for reproducibility
    seed = 1024
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_dataset = OceanDataset("/home/bire/nobackup/mitgcm/baroclinic_gyre/run3/")
    train_datasampler = BatchSampler(train_dataset, batch_size=2, drop_last=True)
    train_dataloader = DataLoader(train_dataset, batch_sampler=train_datasampler)

    validation_dataset = OceanDataset("/home/bire/nobackup/mitgcm/baroclinic_gyre/run3/")
    validation_datasampler = BatchSampler(validation_dataset, batch_size=2, drop_last=True)
    validation_dataloader = DataLoader(validation_dataset, batch_sampler=validation_datasampler)

    # train_dataset = ERA5(split="train", check_data=True)
    # train_datasampler = DistributedSampler(train_dataset, shuffle=True)
    # train_dataloader = DataLoader(train_dataset, args.batch_size, sampler=train_datasampler, num_workers=8, pin_memory=True, drop_last=True)

    # val_dataset = ERA5(split="val", check_data=True)
    # val_datasampler = DistributedSampler(val_dataset, shuffle=False)
    # val_dataloader = DataLoader(val_dataset, args.batch_size, sampler=val_datasampler, num_workers=8, pin_memory=True, drop_last=False)

    model = AFNONet(img_size=[h, w], in_chans=x_c, out_chans=y_c, norm_layer=partial(nn.LayerNorm, eps=1e-6))

    optimizer = torch.optim.AdamW(lr=lr, betas=(0.9, 0.95))
    # loss_scaler = torch.cpu.amp.GradScaler(enabled=True)
    criterion = nn.MSELoss()

    for epoch in range(start_epoch, epochs):

        train_one_epoch(epoch, start_step, model, criterion, train_dataloader, optimizer, loss_scaler, lr_scheduler, min_loss)
        start_step = 0
        lr_scheduler.step(epoch)

        train_loss = single_step_evaluate(train_dataloader, model, criterion)
        val_loss = single_step_evaluate(val_dataloader, model, criterion)

        if rank == 0 and local_rank == 0:
            print(f"Epoch {epoch} | Train loss: {train_loss:.6f}, Val loss: {val_loss:.6f}")
            if val_loss < min_loss:
                min_loss = val_loss
                save_model(model, path=SAVE_PATH / 'backbone.pt', only_model=True)
            save_model(model, epoch+1, 0, optimizer, lr_scheduler, loss_scaler, min_loss, SAVE_PATH / 'pretrain_latest.pt')


if __name__ == '__main__':
    # ngpus = torch.cuda.device_count()
    # torch.multiprocessing.spawn(main, args=(), nprocs=ngpus)
