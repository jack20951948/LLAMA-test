from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
import os

output_dir = "results/llama2/final_checkpoint"
model_name = "meta-llama/Llama-2-7b-hf"

model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map={"": 0})
model = model.merge_and_unload()

output_merged_dir = "results/llama2/final_merged_checkpoint"
os.makedirs(output_merged_dir, exist_ok=True)
model.save_pretrained(output_merged_dir, safe_serialization=True)

# save tokenizer for easy inference
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(output_merged_dir)