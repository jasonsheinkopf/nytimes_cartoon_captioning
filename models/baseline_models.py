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
    AutoConfig
)

# import bitsandbytes
from peft import LoraConfig, get_peft_model


def blip2_quant(cfg):
    quantization_config = BitsAndBytesConfig(load_in_8_bit=True)
    config = AutoConfig.from_pretrained(cfg.MODEL.BASE_MODEL)
    config.attention_probs_dropout_prob = cfg.MODEL.DROPOUT_RATE
    base_model = Blip2ForConditionalGeneration.from_pretrained(
        cfg.MODEL.BASE_MODEL,
        device_map="auto",
        quantization_config=quantization_config,
        config=config)

    return base_model


def opt_2_7(cfg):
    """
    Unmodified BLIP2
    """
    # get quantized base model
    base_model = blip2_quant(cfg)

    # add LoRA adapters for all layers
    model = get_peft_model(base_model,
                           LoraConfig(
                               r=16,
                               lora_alpha=32,
                               lora_dropout=cfg.MODEL.LORA_DROPOUT,
                               bias="none",
                               target_modules=[
                                #    "qformer",
                                    "q_proj", 
                                    "k_proj",
                                    # "v_proj",
                                    # "out_proj",
                                    "language_projection"
                                    ]
                                )
                            )
    
    model.print_trainable_parameters()

    return model


def opt_2_7_qlang(cfg):
    """
    Unmodified BLIP2
    """
    # get quantized base model
    base_model = blip2_quant(cfg)

    # add LoRA adapters for all layers
    model = get_peft_model(base_model,
                           LoraConfig(
                               r=16,
                               lora_alpha=32,
                               lora_dropout=cfg.MODEL.LORA_DROPOUT,
                               bias="none",
                               target_modules=[
                                   "qformer",
                                    # "q_proj", 
                                    # "k_proj",
                                    # "v_proj",
                                    # "out_proj",
                                    "language_projection"
                                    ]
                                )
                            )
    
    model.print_trainable_parameters()

    return model
