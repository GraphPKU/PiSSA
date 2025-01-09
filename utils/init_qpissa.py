import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
import bitsandbytes as bnb
from tqdm import tqdm

# python utils/init_qpissa.py --base_model_dir meta-llama/Llama-2-7b-hf/ --output_path llama-2-7b-pissa-4bit-r128-iter5 --iter 5

parser = argparse.ArgumentParser(description="Initializing QPiSSA.")
parser.add_argument("--base_model_dir", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)
parser.add_argument("--rank", type=int, default=128)
parser.add_argument("--iter", type=int, default=1)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument('--target_modules', nargs='+', help='', required=True)
args = parser.parse_args()

def quantize_and_dequantized(weight):
    device = weight.device
    weight_nf4 = bnb.nn.Params4bit(weight.to("cpu"), requires_grad=False, compress_statistics=False, quant_type="nf4")
    weight_nf4 = weight_nf4.to(device)
    weight_dequantized = bnb.functional.dequantize_4bit(
        weight_nf4.data, weight_nf4.quant_state
    ).to(torch.float32)
    return weight_nf4, weight_dequantized

@torch.no_grad()
def pissa_quant(weight, r=64, niter=5):
    res = weight.to(torch.float32)
    for i in range(niter):
        U, S, Vh = torch.linalg.svd(res, full_matrices=False)
        L = U @ (torch.sqrt(torch.diag(S)[:, :r]))
        R = torch.sqrt(torch.diag(S)[:r, :]) @ Vh
        res = weight - L @ R
        weight_nf4, weight_dequantized = quantize_and_dequantized(res)
        res = weight - weight_dequantized

    return weight_nf4, weight_dequantized, R, L


base_model = AutoModelForCausalLM.from_pretrained(args.base_model_dir, device_map=args.device)
tokenizer = AutoTokenizer.from_pretrained(args.base_model_dir)
lora_config = LoraConfig(
    r=args.rank,
    lora_alpha=args.rank,
    target_modules=args.target_modules,
    task_type="CAUSAL_LM",
)
peft_model = get_peft_model(base_model, peft_config=lora_config)
state_dict = {}
for key, value in tqdm(peft_model.state_dict().items()):
    if "base_layer" in key:
        base_layer_in_4bits, base_layer, lora_A, lora_B = pissa_quant(value, args.rank, args.iter)
        state_dict[key] = base_layer.to('cpu')
        state_dict[key.replace("base_layer", "lora_A.default")] = lora_A.to('cpu')
        state_dict[key.replace("base_layer", "lora_B.default")] = lora_B.to('cpu')

print(peft_model.load_state_dict(state_dict, strict=False))
peft_model.save_pretrained(f"{args.output_path}/pissa_init")
peft_model = peft_model.unload()
peft_model.save_pretrained(args.output_path)
tokenizer.save_pretrained(args.output_path)