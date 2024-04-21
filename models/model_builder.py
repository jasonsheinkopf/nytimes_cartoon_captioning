"""
Different architectures of BLIP2 models.
"""

from .build import MODEL_REGISTRY
from .baseline_models import opt_2_7, blip2_quant, opt_1_3, opt_2_7_identity, opt_1_3_qformer_projection

# register function
MODEL_REGISTRY.register(opt_2_7)
MODEL_REGISTRY.register(blip2_quant)
MODEL_REGISTRY.register(opt_2_7_identity)
MODEL_REGISTRY.register(opt_1_3)
MODEL_REGISTRY.register(opt_1_3_qformer_projection)
