# the structure of this script -s based on https://github.com/epic-kitchens/epic-kitchens-slowfast

"""
Test BLIP2 image captioning model.
"""

import numpy as np
import torch
import sys, os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_dir, '..', '..', 'BLIP2CAP')

sys.path.append(project_path)
from models.build import build_model
from data.nytimes import build_data_loader
from metrics.evaluation import evaluate_captions


def perform_test(test_loader, model, num_iterations):
    all_gen_ids = []
    all_input_ids = []
    with torch.no_grad():
        test_loss = 0
        model.eval()
        for idx, batch in enumerate(test_loader):
            if idx > num_iterations:
                break
            # get ground truth caption for item
            input_ids = batch.pop('input_ids').to(model.device, torch.long)
            # get ground truth image for item
            pixel_values = batch.pop('pixel_values').to(model.device, torch.float32)

            # generate outputs from model
            outputs = model(input_ids=input_ids,
                            pixel_values=pixel_values,
                            labels=input_ids)
            
            # add to cumulative test loss
            test_loss += outputs.loss.item()

            # get generated ids from model
            gen_ids = model.generate(pixel_values, max_length=50)

            # pad generated IDs to ensure consistent size
            max_len = max(len(ids) for ids in gen_ids)
            padded_gen_ids = [ids + [1] * (max_len - len(ids)) for ids in gen_ids]

            # accumulate predictions and ground truth captions
            all_gen_ids.append(padded_gen_ids)
            all_input_ids.append(input_ids)
            
        # average test loss
        test_loss /= len(test_loader)

        # concatenate preds and ground truth
        all_gen_ids = torch.cat(all_gen_ids)
        all_input_ids = torch.cat(all_input_ids)

        # perform evaluation on captioning metrics
        evaluate_captions(all_input_ids, all_gen_ids)

        return test_loss


def test(cfg):
    """
    Test model on image captioning dataset.
    Args:
        cfg (CfgNode): configs
    """
    print('Testing with config:')
    print(cfg)

    np.random.seed(cfg.RNG)
    torch.manual_seed(cfg.RNG)

    model = build_model(cfg)

    # load data
    _, test_loader = build_data_loader(cfg)
    print(f'Testing model for {len(test_loader)} iterations.')
    
    # test on entire dataset
    test_loss = perform_test(test_loader, model, cfg.TEST.NUM_ITER)
    print(f'Test loss: {test_loss}')