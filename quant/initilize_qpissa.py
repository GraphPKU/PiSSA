import torch
from copy import deepcopy
import argparse
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig
from peft.utils.loftq_utils import NFQuantizer
import bitsandbytes as bnb
from tqdm import tqdm

# python init.py --base_model_dir meta-llama/Llama-2-7b-hf/ --output_path llama-2-7b-pissa-4bit-r128-iter5 --iter 5

parser = argparse.ArgumentParser(description="Generate any bit normal float PiSSA.")
parser.add_argument(
    "--base_model_dir",
    type=str,
    required=True,
)
parser.add_argument(
    "--output_path",
    type=str,
    required=True,
)
parser.add_argument(
    "--rank",
    type=int,
    default=128,
)
parser.add_argument(
    "--iter",
    type=int,
    default=1,
)
parser.add_argument(
    "--bits",
    type=int,
    default=4,
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda",
)
args = parser.parse_args()


class QDQ:
    def __init__(self, bits, device="cuda") -> None:
        self.bits = bits
        self.device = device
        self.quantizer = (
            NFQuantizer(num_bits=bits, device=device, method="normal", block_size=64)
            if bits != 4
            else None
        )

    def quantize_and_dequantized(self, weight):
        if self.bits == 4:
            weight_4bits = bnb.nn.Params4bit(
                weight, requires_grad=False, compress_statistics=False, quant_type="nf4"
            ).to(self.device)
            weight_dequantized = bnb.functional.dequantize_4bit(
                weight_4bits.data, weight_4bits.quant_state
            ).to(torch.float32)
        else:
            weight_4bits, max_abs, shape = self.quantizer.quantize_block(weight)
            weight_dequantized = self.quantizer.dequantize_block(
                weight_4bits, max_abs, shape
            )
        return weight_dequantized


@torch.no_grad()
def pissa_quant(weight, qdq, r=64, niter=5):
    res = deepcopy(weight).to(torch.float32)
    for i in range(niter):
        U, S, Vh = torch.linalg.svd(res, full_matrices=False)
        L = U @ (torch.sqrt(torch.diag(S)[:, :r]))
        R = torch.sqrt(torch.diag(S)[:r, :]) @ Vh
        res = weight - L @ R
        weight_dequantized = qdq.quantize_and_dequantized(res)
        res = weight - weight_dequantized

    return weight_dequantized, R, L


base_model = AutoModelForCausalLM.from_pretrained(
    args.base_model_dir, torch_dtype=torch.float16, device_map=args.device
)
lora_config = LoraConfig(
    r=args.rank,
    lora_alpha=args.rank,
    target_modules=[
        "q_proj",
        "o_proj",
        "k_proj",
        "v_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    task_type="CAUSAL_LM",
)
peft_model = get_peft_model(base_model, peft_config=lora_config)

qdq = QDQ(args.bits, args.device)

state_dict = {}
for key, value in tqdm(peft_model.state_dict().items()):
    if "base_layer" in key:
        base_layer, lora_A, lora_B = pissa_quant(value, qdq, args.rank, args.iter)
        state_dict[key] = base_layer
        state_dict[key.replace("base_layer", "lora_A.default")] = lora_A
        state_dict[key.replace("base_layer", "lora_B.default")] = lora_B

print(peft_model.load_state_dict(state_dict, strict=False))
peft_model.save_pretrained(f"{args.output_path}/pissa_init")
peft_model = peft_model.unload()
peft_model.save_pretrained(args.output_path)
