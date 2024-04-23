# the structure of this script is based on https://github.com/epic-kitchens/epic-kitchens-slowfast

"""
Train BLIP2 image captioning model.
"""

import torch
import sys, os
from tqdm.auto import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_dir, '..', '..', 'BLIP2CAP')
sys.path.append(project_path)

from test_net import test
import math


def log_data(train_loss, test_loss, metrics, cur_epoch, cfg, wandb):
    print(f'Epoch: {cur_epoch}, Train Loss: {train_loss}, Test Loss: {test_loss}')
    # only log if wandb has been initialized
    if wandb.run is not None:
        wandb.log({
            'test_loss': test_loss,
            'bleu': metrics['bleu'],
            'rouge_1_f1': metrics['rouge_1_f1'],
            'meteor': metrics['meteor']
        })


def train_epoch(train_loader, model, optimizer, cur_epoch, cfg, wandb):
    """
    Train on image captioning dataset for one epoch.
    Args:
        train_loader (loader)
        model (model)
        optimizer (optim)
        cur_epoch (int)
    """
    # train mode
    model.train()

    # init loss
    train_loss = 0

    num_batches = len(train_loader) if cfg.TRAIN.NUM_BATCHES == -1 else cfg.TRAIN.NUM_BATCHES

    # iterate over training dataloader
    for idx, batch in enumerate(train_loader):
        if idx >= num_batches:
            break
        input_ids = batch.pop('input_ids').to(model.device, torch.long)
        pixel_values = batch.pop('pixel_values').to(model.device, torch.float32)

        # generate outputs
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            labels=input_ids)
        
        # get loss
        loss = outputs.loss

        # log iteration train loss
        if wandb is not None:
            wandb.log({
                'train_loss': loss
            })
        
        # update accumulated batch loss
        train_loss += outputs.loss.item()

        # perform backprop
        loss.backward()

        # update optimizer
        optimizer.step()
        optimizer.zero_grad()

        # Print loss approximately 10 times per batch
        if idx % math.ceil(num_batches / 10) == 0:
            print(f' Batch {idx} train loss: {loss}')
    
    # find average loss
    # print(f'Cumulative train loss on {num_batches} batches: {train_loss}')
    train_loss /= num_batches

    return train_loss


def train(cfg, model, train_loader, test_loader, processor, wandb):
    """
    Train model on image captioning dataset.
    Args:
        cfg (CfgNode): configs
    """
    print('Training with config:')
    print(cfg)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.MODEL.LR, weight_decay=cfg.MODEL.L2_REG)

    # get first sample from val set
    sample_datapoint = next(iter(test_loader))

    print(f"Target caption: {test_loader['image_description'][IDX]}")
    example_inputs = processor(images=example_image, return_tensors="pt").to(device, torch.float16)
    example_pixel_values = example_inputs.pixel_values

    for cur_epoch in range(cfg.TRAIN.EPOCHS):
        print(f'\n==================================================')
        print(f'Training epoch: {cur_epoch}')
        print(f'==================================================\n')
        # consider shuffling dataset
        # train for one epoch
        train_loss = train_epoch(train_loader, model, optimizer, cur_epoch, cfg, wandb)

        print(f'\nEpoch {cur_epoch} training loss: {train_loss}')

        # test on validation set
        test_loss, metrics = test(cfg, model, test_loader, processor, wandb, cur_epoch)

        # generate data logs
        log_data(train_loss, test_loss, metrics, cur_epoch, cfg, wandb)
    
