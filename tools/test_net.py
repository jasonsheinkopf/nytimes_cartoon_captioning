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


def perform_test(test_loader, model, cfg, processor):
    all_gen_ids_list = []
    all_input_ids_list = []
    with torch.no_grad():
        test_loss = 0
        model.eval()
        num_batches = len(test_loader) if cfg.TEST.NUM_BATCHES == -1 else cfg.TEST.NUM_BATCHES

        for idx, batch in enumerate(test_loader):
            if idx >= num_batches:
                break
            # get ground truth caption for item
            input_ids = batch.pop('input_ids').to(model.device, torch.long)
            # get ground truth image for item
            pixel_values = batch.pop('pixel_values').to(model.device, torch.float32)
            # get generated ids from model for evaluation
            gen_ids = model.generate(pixel_values, max_length=50)

            print(f'{len(input_ids)=} | {len(gen_ids)=}')
            print(f'{input_ids.type=} | f{gen_ids.type=}')
            print(f'{input_ids.shape=} | f{gen_ids.shape=}')
            # generate outputs from model
            outputs = model(input_ids=gen_ids,
                            pixel_values=pixel_values,
                            labels=input_ids)
            
            # add to cumulative test loss
            test_loss += outputs.loss.item()

            # concatenate list of batch ids to accumulated list
            all_gen_ids_list += gen_ids.tolist()
            all_input_ids_list += input_ids.tolist()
            
        # average test loss
        test_loss /= num_batches

        # perform evaluation on captioning metrics
        metrics = evaluate_captions(all_input_ids_list, all_gen_ids_list)

        # infer on single sample
        _ = infer(test_loader, model, processor, 1, cfg)

        return test_loss, metrics


def test(cfg, model, test_loader, processor):
    """
    Test model on image captioning dataset.
    Args:
        cfg (CfgNode): configs
    """
    
    # test on dataset
    test_loss, metrics = perform_test(test_loader, model, cfg, processor)

    return test_loss, metrics
