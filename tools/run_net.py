# the structure of this script is based on https://github.com/epic-kitchens/epic-kitchens-slowfast

import argparse
import sys
import torch
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_dir, '..', '..', 'blip2cap')
sys.path.append(project_path)

from data.nytimes import build_data_loader
from models.build import build_model
from configs.defaults import get_cfg
from test_net import test
from train_net import train
import numpy as np
from metrics.evaluation import infer
import wandb


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
    print('Running with config')
    print(cfg)

    # Access WANDB_KEY environment variable with default value None
    wandb_key = os.environ.get("WANDB_API_KEY", None)

    # if wandb_key is None:
    #     print('Wandb_key not detected. Net will not be logged. To add your key, run this line before the script.')
    #     print('os.environ["WANDB_API_KEY"] = input("Please enter your WandB API key: ")')

    np.random.seed(cfg.RNG)
    torch.manual_seed(cfg.RNG)

    if wandb_key is not None:
        wandb.login(key=wandb_key)
    
    # only log run in train mode if a key is provided
    if cfg.TRAIN.ENABLE:
        # cast config to dict
        cfg_dict = dict(cfg)
        wandb.init(
            project=cfg.TRAIN.WANDB_PROJECT,
            entity=cfg.TRAIN.WANDB_ENTITY,
            name=cfg.MODEL.ARCH,
            config=cfg_dict,
            notes=cfg.MODEL.NOTES,
        )

    model = build_model(cfg)

    # load data
    train_loader, test_loader, processor = build_data_loader(cfg)

    if cfg.TRAIN.ENABLE:
        train(cfg, model, train_loader, test_loader, processor, wandb)
        # infer on first X samples and save to wandb after run
        gen_text, metrics = infer(test_loader, model, processor, -1, cfg)

        test_output_text = ""
        for idx, item in enumerate(gen_text):
            test_output_text += f"{idx}: {item}\n"
        print(f'\nFinal captions\n{test_output_text}')

        if wandb_key is not None:
            # save the output file to wandb run dir
            wandb_run_dir = wandb.run.dir
            gens_path = os.path.join(wandb_run_dir, f'{wandb.run.id}_gen_captions.txt')
            # write file to disk
            with open(gens_path, 'w') as f:
                f.write(test_output_text)
            # save to wandb run
            wandb.save(gens_path, base_path=wandb_run_dir)
        else:
            print(test_output_text)

    else:
        gen_text, metrics = test(cfg, model, test_loader, processor)
        print('\nMetrics on test set')
        print(metrics)
        


if __name__ == "__main__":
    main()
