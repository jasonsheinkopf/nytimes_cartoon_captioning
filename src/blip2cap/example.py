from transformers import (
    Blip2VisionConfig,
    Blip2QFormerConfig,
    OPTConfig,
    Blip2Config,
    Blip2ForConditionalGeneration,
    #PhiConfig,
    Blip2Model
)

base_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", 
    device_map="auto", load_in_8bit=True)