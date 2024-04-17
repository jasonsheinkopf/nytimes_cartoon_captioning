"""
Different architectures of BLIP2 models.
"""

from .build import MODEL_REGISTRY
from .baseline_models import opt_2_7, blip2_quant, opt_2_7_qlang

# register function
MODEL_REGISTRY.register(opt_2_7)
MODEL_REGISTRY.register(opt_2_7_qlang)
MODEL_REGISTRY.register(blip2_quant)
