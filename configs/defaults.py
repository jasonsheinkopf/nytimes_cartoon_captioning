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

# Resume training from checkpoint
_C.TRAIN.AUTO_RESUME = True

################################
# Testing
################################

_C.TEST = CfgNode()

# dataset
_C.TEST.DATASET = "nytimes"

_C.TEST.BATCH_SIZE = 8

################################
# Model
################################

_C.MODEL = CfgNode()

# Path to checkpoint path
_C.MODEL.CHECKPOINT_FILE_PATH = ""

# architecture
_C.MODEL.ARCH = ""

# dropout rate
_C.MODEL.DROPOUT_RATE = 0.5

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

def get_cfg():
    """
    Return copy of config.
    """
    return _C.clone()
