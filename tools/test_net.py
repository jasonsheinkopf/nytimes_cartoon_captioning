# the structure of this script -s based on https://github.com/epic-kitchens/epic-kitchens-slowfast

"""
Test BLIP2 image captioning model.
"""

import torch
import sys, os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_dir, '..', '..', 'BLIP2CAP')

sys.path.append(project_path)

from metrics.evaluation import evaluate_captions, infer


def test(cfg, model, test_loader, processor):
    """
    Test model on image captioning dataset.
    Args:
        cfg (CfgNode): configs
    """
    with torch.no_grad():
        test_loss = 0
        model.eval()
        num_batches = len(test_loader) if cfg.TEST.NUM_BATCHES == -1 else cfg.TEST.NUM_BATCHES
        print(f'\n==================================================')
        print(f'Testing.')
        print(f'==================================================\n')

        for idx, batch in enumerate(test_loader):
            if idx >= num_batches:
                break
            # get ground truth caption for item
            input_ids = batch.pop('input_ids').to(model.device, torch.long)
            # get ground truth image for item
            pixel_values = batch.pop('pixel_values').to(model.device, torch.float32)

            # forward pass through model
            outputs = model(input_ids=input_ids,
                            pixel_values=pixel_values,
                            labels=input_ids)
            
            # add to cumulative test loss
            test_loss += outputs.loss.item()

        # average test loss
        test_loss /= num_batches
        print(f'\nTest loss: {test_loss}')

        # infer on single sample
        _, metrics = infer(test_loader, model, processor, 1, cfg)

        return test_loss, metrics

