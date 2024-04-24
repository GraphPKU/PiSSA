import torch
import argparse
from transformers import AutoModelForCausalLM
from peft.utils.loftq_utils import NFQuantizer
import bitsandbytes as bnb
from tqdm import tqdm

# python init_qlora.py --base_model_dir meta-llama/Llama-2-7b-hf/ --output_path llama-2-7b-qlora-2bit --bits 2

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


base_model = AutoModelForCausalLM.from_pretrained(
    args.base_model_dir, torch_dtype=torch.float16, device_map=args.device
)
qdq = QDQ(args.bits, args.device)

state_dict = {}
with torch.no_grad():
    for key, value in tqdm(base_model.state_dict().items()):
        if "_proj" in key:
            weight_dequantized = qdq.quantize_and_dequantized(value)
            state_dict[key] = weight_dequantized

print(base_model.load_state_dict(state_dict, strict=False))
base_model.save_pretrained(args.output_path)
