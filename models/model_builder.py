"""
Different architectures of BLIP2 models.
"""

from .build import MODEL_REGISTRY
from .baseline_models import opt_2_7, blip2_quant, opt_1_3, opt_2_7_identity, opt_1_3_qformer_projection, opt_1_3_colab, opt_1_3_new_linear_layer,opt_1_3_svd, opt_2_7_no_peft

# register function
MODEL_REGISTRY.register(opt_2_7)
MODEL_REGISTRY.register(opt_2_7_no_peft)
MODEL_REGISTRY.register(opt_2_7_identity)
MODEL_REGISTRY.register(opt_1_3)
MODEL_REGISTRY.register(opt_1_3_new_linear_layer)
MODEL_REGISTRY.register(opt_1_3_svd)
MODEL_REGISTRY.register(opt_1_3_qformer_projection)
MODEL_REGISTRY.register(opt_1_3_colab)
