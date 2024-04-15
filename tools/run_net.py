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
    print(cfg)

    print(f'Running with config')
    print(cfg)

    # Access WANDB_KEY environment variable with default value None
    wandb_key = os.environ.get("WANDB_API_KEY", None)

    if wandb_key is None:
        print('Wandb_key not detected. Net will not be logged. To add your key, run this line before the script.')
        print('os.environ["WANDB_API_KEY"] = input("Please enter your WandB API key: ")')

    np.random.seed(cfg.RNG)
    torch.manual_seed(cfg.RNG)

    if wandb_key is not None:
        wandb.login(key=wandb_key)
        wandb.init(
            project='blip2cap',
            entity=cfg.TRAIN.WANDB_ENTITY,
            name=cfg.MODEL.ARCH,
            config={
                'Train Batch Size': cfg.TRAIN.BATCH_SIZE,
                'Test Batch Size': cfg.TEST.BATCH_SIZE,
                'Dropout Rate': cfg.MODEL.DROPOUT_RATE,
                'Dataset': cfg.DATA.DATASET
            }
        )

    model = build_model(cfg)

    # load data
    train_loader, test_loader, processor = build_data_loader(cfg)


    if cfg.TRAIN.ENABLE:
        train(cfg, model, train_loader, test_loader, processor, wandb)
        # infer on first X samples and save to wandb after run
        gen_text = infer(test_loader, model, processor, -1, cfg)

        print('Generated text list test dataset')
        for idx, item in enumerate(gen_text):
            print(f'{idx}: {item}')

        # create line separated single text string of all gen responses
        test_output_text = '/n'.join(gen_text)

        # create wandb artifact and log directly from memory
        artifact = wandb.Artifact("test_genearations", type="dataset")
        artifact.add_text("test_generations.txt", test_output_text)
        wandb.log_artifact(artifact)

    else:
        test(cfg, model, test_loader, processor)


if __name__ == "__main__":
    main()
