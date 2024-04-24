from transformers import AutoModelForCausalLM
from peft import PeftModel
import torch
import argparse
parser = argparse.ArgumentParser(description="Calcualte the quantize error of a peft model.")
parser.add_argument(
        "--base_model_dir",
        type=str,
        required=True,
    )
parser.add_argument(
        "--quantize_model_dir",
        type=str,
        required=True,
    )
parser.add_argument(
        "--adapter_dir",
        type=str,
        default=None,
    )
args = parser.parse_args()
base_model = AutoModelForCausalLM.from_pretrained(args.base_model_dir)
quantize_model = AutoModelForCausalLM.from_pretrained(args.quantize_model_dir)
if args.adapter_dir:
    peft_model = PeftModel.from_pretrained(quantize_model, args.adapter_dir)
    quantize_model = peft_model.merge_and_unload()

error_singular_value_dict={}

for key, value in base_model.state_dict().items():
    if "_proj" in key:
        error_singular_value_dict[key] = torch.linalg.svdvals(value - quantize_model.state_dict()[key])
        print(key, error_singular_value_dict[key].sum())

torch.save(error_singular_value_dict, f"{args.quantize_model_dir}/error_singular_value_dict.pt")