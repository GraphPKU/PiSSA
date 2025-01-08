import torch
import argparse
import bitsandbytes as bnb
from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM

parser = argparse.ArgumentParser(
    description="Calculate the quantization error of NF4 model."
)
parser.add_argument(
    "--base_model_path",
    type=str,
    required=True,
)
parser.add_argument(
    "--quant_model_path",
    type=str,
    required=True,
)
parser.add_argument(
    "--output_path",
    type=str,
    required=True,
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda",
)
args = parser.parse_args()


residual_model = AutoModelForCausalLM.from_pretrained(
    args.base_model_path, torch_dtype=torch.bfloat16, device_map=args.device
)
quant_model = AutoModelForCausalLM.from_pretrained(
    args.quant_model_path, low_cpu_mem_usage=True
)

with torch.no_grad():
    for name, param in quant_model.named_parameters():
        if "_proj" in name:
            W = residual_model.get_parameter(name)
            W.data = bnb.functional.dequantize_4bit(param.data, param.quant_state).to(torch.bfloat16).cpu()


residual_model.save_pretrained(args.output_path)