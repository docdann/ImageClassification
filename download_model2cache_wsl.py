from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

model_id = 'Ertugrul/Qwen2-VL-7B-Captioner-Relaxed'
Qwen2VLForConditionalGeneration.from_pretrained(model_id, torch_dtype='bfloat16', device_map='auto')
AutoProcessor.from_pretrained(model_id)
