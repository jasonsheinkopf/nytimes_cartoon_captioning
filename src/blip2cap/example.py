from transformers import (
    Blip2VisionConfig,
    Blip2QFormerConfig,
    OPTConfig,
    Blip2Config,
    Blip2ForConditionalGeneration,
    #PhiConfig,
    Blip2Model
)
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from datasets import load_dataset 
from transformers import (
    Blip2VisionConfig,
    Blip2QFormerConfig,
    OPTConfig,
    Blip2Config,
    Blip2ForConditionalGeneration,
    #PhiConfig,
    Blip2Model
)
from transformers import AutoProcessor, AutoTokenizer
import bitsandbytes
from peft import LoraConfig, get_peft_model
import torch.nn as nn

def get_model():
    base_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", 
        device_map="auto", load_in_8bit=True)

    original_language_projection = base_model.language_projection

    identity = bitsandbytes.nn.Linear8bitLt(
        original_language_projection.out_features, original_language_projection.out_features, bias=False)
    print(f"original_language_projection: {(original_language_projection.in_features, original_language_projection.out_features)}")

    base_model.language_projection = nn.Sequential(
        original_language_projection, # "language_projection.0",
        nn.ReLU(),
        identity.to(base_model.device) #"language_projection.2"
    )

    with torch.no_grad():
        identity.weight.fill_(0)
        identity.weight += (torch.eye(original_language_projection.out_features).to(base_model.device)
        #Corrupt the identity matrix with noise to mimic initialization state.
            + ((1/3.5) * (torch.randn((original_language_projection.out_features, original_language_projection.out_features)))).char().to(base_model.device)
        )
        print(original_language_projection.weight)
        print(identity.weight)


    model = get_peft_model(base_model, LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            #"q_proj", 
            # "k_proj", 
            "language_projection.0",
            #"language_projection.2"
        ],
    ))

    model.print_trainable_parameters()

    return model
    