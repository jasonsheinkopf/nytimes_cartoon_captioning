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


def test(cfg):
    """
    Test model on image captioning dataset.
    Args:
        cfg (CfgNode): configs
    """
    np.random.seed(cfg.RNG)
    torch.manual_seed(cfg.RNG)

    print(f'Loading model: {cfg.MODEL.ARCH}')
    model = build_model(cfg)
    
    print(model)