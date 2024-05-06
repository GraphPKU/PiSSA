import torch
import argparse
import bitsandbytes as bnb
from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM
from peft.tuners.lora import Linear4bit
from collections import defaultdict

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
    "--adapter_name",
    type=str,
    default="pissa_init",
)
parser.add_argument(
    "--scaling",
    type=int,
    default=1,
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda",
)
args = parser.parse_args()


@torch.no_grad()
def quantization_error_reduction_ratio(W, W_res, lora_A=None, lora_B=None, scaling=1):
    W_nf4 = bnb.nn.Params4bit(
        W, requires_grad=False, compress_statistics=False, quant_type="nf4"
    ).to(W_res.device)
    W_dequantized = bnb.functional.dequantize_4bit(W_nf4.data, W_nf4.quant_state).to(
        torch.float32
    )
    W_res_dequantized = bnb.functional.dequantize_4bit(
        W_res.data, W_res.quant_state
    ).to(torch.float32)
    if lora_A is not None and lora_B is not None:
        W_res_dequantized += scaling * lora_B @ lora_A
    return (
        1 - torch.linalg.svdvals(W - W_res_dequantized).sum()
        / torch.linalg.svdvals(W - W_dequantized).sum()
    ).to("cpu")


base_model = AutoModelForCausalLM.from_pretrained(
    args.base_model_path, torch_dtype=torch.bfloat16, device_map=args.device
)
quant_model = AutoModelForCausalLM.from_pretrained(
    args.quant_model_path, low_cpu_mem_usage=True
)

peft_model = PeftModel.from_pretrained(
    quant_model,
    args.quant_model_path,
    subfolder=args.adapter_name,
)
error_singular_value_dict = {}
sum_of_error = defaultdict(list)
for name, module in peft_model.named_modules():
    if isinstance(module, Linear4bit):
        W = base_model.get_parameter(name.replace("base_model.model.", "") + ".weight")
        W_res = module.base_layer.weight
        lora_A = module.lora_A["default"].weight
        lora_B = module.lora_B["default"].weight
        error_singular_value_dict[name] = quantization_error_reduction_ratio(
            W.to("cuda"), W_res, lora_A, lora_B
        )
        print(name, error_singular_value_dict[name])
        sum_of_error[name.split(".")[-1]].append(error_singular_value_dict[name])

for key, value in sum_of_error.items():
    print(key, torch.Tensor(value).mean())
