# the structure of this script is based on https://github.com/epic-kitchens/epic-kitchens-slowfast

"""
Train BLIP2 image captioning model.
"""

import numpy as np
import torch
import sys, os
from tqdm.auto import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_dir, '..', '..', 'BLIP2CAP')

sys.path.append(project_path)
from models.build import build_model
from data.nytimes import build_data_loader

def train_epoch(train_loader, model, optimizer, cur_epoch):
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
        input_ids = batch.pop('input_ids').to(model.device, torch.float32)
        pixel_values = batch.pop('pixel_values').to(model.device, torch.float32)

        # generate outputs
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            labels=input_ids)
        
        # get loss
        loss = outputs.loss
        
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


def train(cfg):
    """
    Train model on image captioning dataset.
    Args:
        cfg (CfgNode): configs
    """
    print('Training with config:')
    print(cfg)

    np.random.seed(cfg.RNG)
    torch.manual_seed(cfg.RNG)

    model = build_model(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    start_epoch = 0

    # load data
    train_loader, test_loader = build_data_loader(cfg)
    print(f'Start epoch: {start_epoch}')

    # create meters
    train_meter = TrainMeter()

    for idx, cur_epoch in enumerate(range(cfg.TRAIN.EPOCHS)):
        # consider shuffling dataset
        # train for one epoch
        train_loss = train_epoch(train_loader, model, optimizer, cur_epoch)

        print(f'Epoch {idx} training loss: {train_loss}')
    
