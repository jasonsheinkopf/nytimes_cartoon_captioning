"""
Different architectures of BLIP2 models.
"""

import torch
import torch.nn as nn

from .build import MODEL_REGISTRY

from transformers import (
    Blip2VisionConfig,
    Blip2QFormerConfig,
    OPTConfig,
    Blip2Config,
    BitsAndBytesConfig,
    Blip2ForConditionalGeneration,
    #PhiConfig,
    Blip2Model
)
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from datasets import load_dataset 
from transformers import AutoProcessor, AutoTokenizer
# import bitsandbytes
from peft import LoraConfig, get_peft_model

def blip2_quant():
    quantization_config = BitsAndBytesConfig(load_in_8_bit=True)
    base_model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        device_map="auto",
        quantization_config=quantization_config)

    return base_model

def opt_2_7(cfg):
    """
    Unmodified BLIP2
    """
    # get quantized base model
    base_model = blip2_quant()

    # add LoRA adapters for all layers
    model = get_peft_model(base_model,
                           LoraConfig(
                               r=16,
                               lora_alpha=32,
                               lora_dropout=0.05,
                               bias="none",
                               target_modules=[
                                #    "qformer",
                                    "q_proj", 
                                    "k_proj",
                                    # "v_proj",
                                    # "out_proj",
                                    # "language_projection"
                                    ]
                                    )
                            )

    model.print_trainable_parameters()
    return model


# register function
MODEL_REGISTRY.register(opt_2_7)
