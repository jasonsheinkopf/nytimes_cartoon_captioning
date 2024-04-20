"""
Different architectures of BLIP2 models.
"""

from .build import MODEL_REGISTRY

from transformers import (
    Blip2VisionConfig,
    Blip2QFormerConfig,
    OPTConfig,
    Blip2Config,
    BitsAndBytesConfig,
    Blip2ForConditionalGeneration,
    #PhiConfig,
    Blip2Model,
    AutoConfig,
    AutoModelForCausalLM
)
import torch
import torch.nn as nn
import bitsandbytes
from peft import LoraConfig, get_peft_model

OPT_1_3_INPUT_DIM = 2048

def blip2_quant(cfg):
    #quantization_config = BitsAndBytesConfig(load_in_8_bit=True)
    config = AutoConfig.from_pretrained(cfg.MODEL.BASE_MODEL)
    config.attention_probs_dropout_prob = cfg.MODEL.DROPOUT_RATE
    base_model = Blip2ForConditionalGeneration.from_pretrained(
        cfg.MODEL.BASE_MODEL,
        device_map="auto",
        #quantization_config=quantization_config,
        load_in_8bit=True,
        config=config)

    return base_model


def opt_2_7(cfg):
    """
    BLIP 2 fill full qformer and language projection layer trained with LoRa
    """
    # get quantized base model
    base_model = blip2_quant(cfg)

    target_modules = [
        'attention.attention.query',
        'attention.attention.key',
        'attention.attention.value',
        'crossattention.output.dense',
        'intermediate_query.dense',
        'output_query.query',
        'language_projection'
    ]

    # add LoRA adapters for all layers
    model = get_peft_model(base_model,
                           LoraConfig(
                               r=cfg.LORA.R,
                               lora_alpha=cfg.LORA.ALPHA,
                               lora_dropout=cfg.LORA.DROPOUT,
                               bias=cfg.LORA.BIAS,
                               target_modules=target_modules
                               ))

    model.print_trainable_parameters()

    return model


def opt_2_7_identity(cfg):
    """
    Provides a starting point for modifying the projection layer(s). A square linear
    hidden layer is appended to the original projection layer and initialized to
    an identity matrix. 
    
    The Relu between the original projection layer and identity
    matrix will slightly "corrupt" the model.
    """
    # get quantized base model
    base_model = blip2_quant(cfg)

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
            #+ ((1/4.5) * (torch.randn((original_language_projection.out_features, original_language_projection.out_features)))).char().to(base_model.device)
        )
        print(original_language_projection.weight)
        print(identity.weight)

    # add LoRA adapters for all layers
    model = get_peft_model(base_model,
                           LoraConfig(
                               r=16,
                               lora_alpha=32,
                               lora_dropout=cfg.MODEL.LORA_DROPOUT,
                               bias="none",
                               target_modules=[
                                    #"language_projection.0",
                                    "language_projection.2"
                                    ]
                                )
                            )
    
    print(model)

    return model
