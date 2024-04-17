import torch
from fvcore.common.registry import Registry

MODEL_REGISTRY = Registry("MODEL")

def build_model(cfg):
    """
    Builds BLIP2 model baseds on arch defined in cfg
    """
    # construct model
    arch = cfg.MODEL.ARCH
    model = MODEL_REGISTRY.get(arch)(cfg)
    
    return model