# the structure of this script is based on https://github.com/epic-kitchens/epic-kitchens-slowfast

import argparse
import sys
import torch
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_dir, '..', '..', 'blip2cap')
print(project_path)
sys.path.append(project_path)

from configs.defaults import get_cfg
from test_net import test
from train_net import train


def parse_args():
    """
    Parse args for training and testing pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Provide BLIP2 captioning training and testing pipeline."
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default="configs/opt_2_7_base.yaml",
        type=str
    )
    parser.add_argument(
        "opts",
        help="See configs/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER
    )
    return parser.parse_args()


def load_config(args):
    """
    Load and initialize configuration based on arguments.
    """
    cfg = get_cfg()

    # Load configs from yaml file and overwrite
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    # Load configs from command line and overwrite
    if args.cfg_file is not None:
        cfg.merge_from_list(args.opts)

    return cfg


def main():
    """
    Spawns train and test.
    """
    args = parse_args()
    cfg = load_config(args)
    print(cfg)

    if cfg.TRAIN.ENABLE:
        train(cfg)
    else:
        test(cfg)


if __name__ == "__main__":
    main()
