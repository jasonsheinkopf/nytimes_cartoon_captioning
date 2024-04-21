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

def opt_2_7_no_peft(cfg):
    """
    Opt 2.7 with vision and language models frozen. No peft.
    """
    # get quantized base model
    base_model = blip2_quant(cfg)

    # Freeze all vision model params
    for name, param in base_model.vision_model.named_parameters():
        # Set requires_grad to False for all parameters
        param.requires_grad = False

    # Freeze language model params
    for name, param in base_model.language_model.named_parameters():
        # Set requires_grad to False for all parameters
        param.requires_grad = False

    # Iterate through the named parameters of the qformer and set grad to true when possible
    for name, param in base_model.qformer.named_parameters():
        # Check if the parameter is of a compatible type
        if param.dtype in [torch.float32, torch.float64, torch.complex64, torch.complex128]:
            # Set requires_grad to True for all compatible parameters
            param.requires_grad = True

    # create new trainable linear layer
    base_model.language_projection = bitsandbytes.nn.Linear8bitLt(
        base_model.language_projection.in_features, base_model.language_projection.out_features).to(base_model.device)
    for name, param in base_model.language_projection.named_parameters():
        # Set requires_grad to True for all compatible parameters
        param.requires_grad = True

    return base_model

def opt_2_7(cfg):
    """
    BLIP 2 full qformer and language projection layer trained with LoRa
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
                                )
                            )
    
    model.print_trainable_parameters()

    return model

def opt_1_3_qformer_projection(cfg):
    """
    Maximal language model replacement of Salesforce/blip2-opt-2.7b with the 
    smaller facebook/opt-1.3b model.
    """
    base_model = blip2_quant(cfg)
    original_language_projection = base_model.language_projection

    opt_1_3_lm = AutoModelForCausalLM.from_pretrained(
        'facebook/opt-1.3b',
        #torch_dtype=torch.float16,
        #torch_dtype=torch.bfloat16,
        trust_remote_code=False,
        #local_files_only=True
        device_map='auto',load_in_8bit=True
        )

    base_model.language_model = opt_1_3_lm
    base_model.language_projection = bitsandbytes.nn.Linear8bitLt(
        original_language_projection.in_features, OPT_1_3_INPUT_DIM).to(base_model.device)
    base_model.post_init()

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
                                )
                            )
    
    model.print_trainable_parameters()
    return model

def opt_1_3(cfg):
    """
    Minimal language model replacement of Salesforce/blip2-opt-2.7b with the 
    smaller facebook/opt-1.3b model.
    """
    base_model = blip2_quant(cfg)
    original_language_projection = base_model.language_projection

    opt_1_3_lm = AutoModelForCausalLM.from_pretrained('facebook/opt-1.3b',
        #torch_dtype=torch.float16,
        #torch_dtype=torch.bfloat16,
        trust_remote_code=False,
        #local_files_only=True
        device_map='auto',load_in_8bit=True

    )
    base_model.language_model = opt_1_3_lm
    base_model.language_projection = bitsandbytes.nn.Linear8bitLt(
        original_language_projection.in_features, OPT_1_3_INPUT_DIM).to(base_model.device)
    base_model.post_init()

    # add LoRA adapters for all layers
    model = get_peft_model(base_model,
                           LoraConfig(
                               r=cfg.LORA.R,
                               lora_alpha=cfg.LORA.ALPHA,
                               lora_dropout=cfg.LORA.DROPOUT,
                               bias=cfg.LORA.BIAS,
                               target_modules=[
                                #    "qformer",
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
                               r=cfg.LORA.R,
                               lora_alpha=cfg.LORA.ALPHA,
                               lora_dropout=cfg.LORA.DROPOUT,
                               bias=cfg.LORA.BIAS,
                               target_modules=[
                                    #"language_projection.0",
                                    "language_projection.2"
                                    ]
                                )
                            )
    
    print(model)

    return model
