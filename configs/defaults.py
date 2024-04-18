from fvcore.common.config import CfgNode

################################
# Config definition
################################

_C = CfgNode()

_C.RNG = 42

################################
# Training
################################

_C.TRAIN = CfgNode()

# train if true
_C.TRAIN.ENABLE = True

# mini batch size
_C.TRAIN.BATCH_SIZE = 32

# Eval on test data after each period
_C.TRAIN.EVAL_PERIOD = 1

# Save checkpoint after each period
_C.TRAIN.CHECKPOINT_PERIOD = 1

# Number of epochs to train for
_C.TRAIN.EPOCHS = 10

# Resume training from checkpoint
_C.TRAIN.AUTO_RESUME = True

# Group or name
_C.TRAIN.WANDB_ENTITY = 'captioneers'

_C.TRAIN.WANDB_PROJECT = 'blip2cap'

# Control number of train batches for testing
_C.TRAIN.NUM_BATCHES = -1

################################
# Testing
################################

_C.TEST = CfgNode()

# dataset
_C.TEST.DATASET = "nytimes"

_C.TEST.BATCH_SIZE = 8

_C.TEST.NUM_BATCHES = 10

################################
# Model
################################

_C.MODEL = CfgNode()

# Path to checkpoint path
_C.MODEL.CHECKPOINT_FILE_PATH = ""

# Base Huggingface model
_C.MODEL.BASE_MODEL = "Salesforce/blip2-opt-2.7b"

# architecture
_C.MODEL.ARCH = ""

# detailed notes
_C.MODEL.NOTES = ""

# dropout rate
_C.MODEL.DROPOUT_RATE = 0.5

# LoRA dropout
_C.MODEL.LORA_DROPOUT = 0.05

# L2 regularization term
_C.MODEL.L2_REG = 0.05

# checkpoint dir
_C.MODEL.CHECKPOINT_DIR = "checkpoints"

# device
_C.MODEL.DEVICE = "cuda"

# output dir
_C.MODEL.OUTPUT_DIR = ""

################################
# Data
################################

_C.DATA = CfgNode()

# Dataset
_C.DATA.DATASET = "mhessel/newyorker_caption_contest"

# Which files
_C.DATA.ANNOTATION = "explanation"

# Which feature to use for captioning
_C.DATA.FEATURE = "image_description"

#  Hugging face processor for image processing and text tokenization
_C.DATA.PROCESSOR = "Salesforce/blip2-opt-2.7b"

#

def get_cfg():
    """
    Return copy of config.
    """
    return _C.clone()
