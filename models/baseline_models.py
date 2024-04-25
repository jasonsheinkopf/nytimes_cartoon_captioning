"""
Different architectures of BLIP2 models.
"""

from .build import MODEL_REGISTRY
import numpy as np

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
    base_model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", 
        device_map="auto", load_in_8bit=True)

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

    # # create new trainable linear layer
    # base_model.language_projection = bitsandbytes.nn.Linear8bitLt(
    #     base_model.language_projection.in_features, base_model.language_projection.out_features).to(base_model.device)
    # for name, param in base_model.language_projection.named_parameters():
    #     # Set requires_grad to True for all compatible parameters
    #     param.requires_grad = True

    # Calculate total number of parameters
    total_params = sum(p.numel() for p in base_model.parameters())
    # Calculate number of trainable parameters
    trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)

    # Calculate percentage of trainable parameters
    percentage_trainable = (trainable_params / total_params) * 100

    print(f"Percentage of trainable parameters: {percentage_trainable:.2f}%")

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

    OPT_1_3_INPUT_DIM = 2048

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


def opt_1_3_colab(cfg):
    base_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", 
    device_map="auto", load_in_8bit=True)
    original_language_projection = base_model.language_projection

    opt_1_3_lm = AutoModelForCausalLM.from_pretrained('facebook/opt-1.3b',
        trust_remote_code=False,
        device_map='auto',load_in_8bit=True

    )
    base_model.language_model = opt_1_3_lm
    base_model.language_projection = bitsandbytes.nn.Linear8bitLt(
        original_language_projection.in_features, 2048).to(base_model.device)
    base_model.post_init()

    model = get_peft_model(base_model, LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            #"q_proj", 
            # "k_proj",
            "language_projection"
            ],
            ))

    model.print_trainable_parameters()

    return model
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

def opt_1_3_new_linear_layer(cfg):
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
        device_map='cuda',load_in_8bit=True

    )
    base_model.language_model = opt_1_3_lm
    new_linear_layer = bitsandbytes.nn.Linear8bitLt(
        original_language_projection.out_features, OPT_1_3_INPUT_DIM).to(base_model.device)
    with torch.no_grad():
        torch.nn.init.normal_(new_linear_layer.weight, mean=0.0, std=1.0, generator=None)
        # new_linear_layer.init
    base_model.language_projection = nn.Sequential(
        original_language_projection, # "language_projection.0",
        # nn.BatchNorm1d(32).to(base_model.device),
        nn.ReLU(),
        new_linear_layer #"language_projection.1"
    )
    print("new_linear_layer", new_linear_layer.parameters)
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
                                    "language_projection.2"
                                    ]
                                )
                            )
    model.print_trainable_parameters()

    # for training the full layer
    # model.print_trainable_parameters()
    # for param in base_model.parameters():
    #     param.requires_grad = False
    # for param in new_linear_layer.parameters():
    #     param.requires_grad = True
        # print (name, param.data)
    # return base_model

    return model

def opt_1_3_svd(cfg):
    base_model = blip2_quant(cfg)
    ORIGINAL_LANGUAGE_PROJECTION_WEIGHTS_FILE = './models/original_language_projection.float32.pt'

    #https://huggingface.co/blog/hf-bitsandbytes-integration#usage
    # base_model_fp32 = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", device_map="auto")
    # torch.save(base_model_fp32.language_projection.state_dict(), ORIGINAL_LANGUAGE_PROJECTION_WEIGHTS_FILE)

    original_language_projection_fp32 = torch.load(ORIGINAL_LANGUAGE_PROJECTION_WEIGHTS_FILE)
    # print(original_language_projection_fp32['weight'])

    lp_weight = original_language_projection_fp32['weight'].cpu().numpy()

    U,S,Vh = np.linalg.svd(lp_weight.T, full_matrices=False) # (U @ np.diag(S)) @ V
    print("U,S,Vh: ", U.shape, S.shape, Vh.shape) #(768, 768) (768,) (768, 2560)
    principal_components = U @ np.diag(S)
    print("U * S = principal components: ", principal_components)
    principal_components_layer_quantized = bitsandbytes.nn.Linear8bitLt(
        principal_components.shape[0], principal_components.shape[1])

    language_reprojection = bitsandbytes.nn.Linear8bitLt(
        principal_components.shape[1], 2048)
    language_reprojection.to(base_model.device)
    print("Initialized language reprojection (quantized): ", language_reprojection.weight)

    principal_components_layer_fp32 = nn.Linear(principal_components.shape[0],principal_components.shape[1])
    with torch.no_grad():
        principal_components_layer_fp32.weight.copy_(torch.tensor(principal_components))
    print("Initialized principal components layer weights (fp32): ", principal_components_layer_fp32.weight)

    principal_components_layer_quantized.load_state_dict(principal_components_layer_fp32.state_dict())
    principal_components_layer_quantized.to(base_model.device) #Performs the quantization. See https://huggingface.co/blog/hf-bitsandbytes-integration#usage 
    print("Initialized principal components layer weights (quantized): ", principal_components_layer_quantized.weight)

    base_model.language_projection = nn.Sequential(
        principal_components_layer_quantized, # "language_projection.0",
        #nn.ReLU(),
        language_reprojection #"language_projection.1"
    )
    opt_1_3_lm = AutoModelForCausalLM.from_pretrained('facebook/opt-1.3b',
        #torch_dtype=torch.float16,
        #torch_dtype=torch.bfloat16,
        trust_remote_code=False,
        device_map='auto',load_in_8bit=True

    )
    base_model.language_model = opt_1_3_lm
    base_model.post_init()

    model = get_peft_model(base_model, LoraConfig(
        r=cfg.LORA.R,
        lora_alpha=cfg.LORA.ALPHA,
        lora_dropout=cfg.LORA.DROPOUT,
        bias=cfg.LORA.BIAS,
        target_modules=[
            "language_projection.0",
            "language_projection.1"
        ],
    ))

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
