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
import wandb

from data.nytimes import build_data_loader


def log_data(train_loss, test_loss, metrics, cur_epoch, cfg, wandb):
    print(f'Epoch: {cur_epoch}, Train Loss: {train_loss}, Test Loss: {test_loss}')
    if wandb is not None:
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

    # iterate over training dataloader
    for idx, batch in tqdm(enumerate(train_loader)):
        if idx > 10:
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
    
    # find average loss
    train_loss /= len(train_loader)

    return train_loss


def train(cfg, model, train_loader, test_loader, processor, wandb):
    """
    Train model on image captioning dataset.
    Args:
        cfg (CfgNode): configs
    """
    print('Training with config:')
    print(cfg)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    start_epoch = 0

    print(f'Start epoch: {start_epoch}')

    for idx, cur_epoch in enumerate(range(cfg.TRAIN.EPOCHS)):
        # consider shuffling dataset
        # train for one epoch
        train_loss = train_epoch(train_loader, model, optimizer, cur_epoch, cfg, wandb)

        print(f'Epoch {idx} training loss: {train_loss}')

        # test on validation set
        test_loss, metrics = test(cfg, model, test_loader, processor)

        # generate data logs
        log_data(train_loss, test_loss, metrics, cur_epoch, cfg, wandb)
    
