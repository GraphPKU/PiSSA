from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import argparse
import torch

parser = argparse.ArgumentParser(description='Merge Adapter to Base Model')
parser.add_argument('--base_model', type=str)
parser.add_argument('--adapter', type=str)
parser.add_argument('--output_path', type=str)
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(args.base_model)
model = PeftModel.from_pretrained(model, args.adapter)
model = model.merge_and_unload()
model.save_pretrained(args.output_path)
tokenizer.save_pretrained(args.output_path)