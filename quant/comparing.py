import torch
qlora_r128_error_singular_value_dict = torch.load("pissa_quant/llama-2-7b-qlora-2bit/error_singular_value_dict.pt")
qlora_r128_error_singular_value_dict = torch.load("pissa_quant/llama-2-7b-qlora-4bit/error_singular_value_dict.pt")
loftq_r128_1iter_error_singular_value_dict = torch.load("pissa_quant/llama-2-7b-loftq-4bit-r128-iter1/error_singular_value_dict.pt")
loftq_r128_5iter_error_singular_value_dict = torch.load("pissa_quant/llama-2-7b-loftq-4bit-r128-iter5/error_singular_value_dict.pt")
pissa_1iter_error_r128_singular_value_dict = torch.load("pissa_quant/llama-2-7b-pissa-4bit-r128-iter1/error_singular_value_dict.pt")
pissa_5iter_error_r128_singular_value_dict = torch.load("pissa_quant/llama-2-7b-pissa-4bit-r128-iter5/error_singular_value_dict.pt")

for proj in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
    print(proj)
    loftq_1iter_err_rate=0
    loftq_5iter_err_rate=0
    pissa_1iter_err_rate=0
    pissa_5iter_err_rate=0
    for key,value in qlora_r128_error_singular_value_dict.items():
        if key.split(".")[-2] == proj:
            loftq_1iter_err_rate += loftq_r128_1iter_error_singular_value_dict[key].sum()/value.sum()
            loftq_5iter_err_rate += loftq_r128_5iter_error_singular_value_dict[key].sum()/value.sum()
            pissa_1iter_err_rate += pissa_1iter_error_r128_singular_value_dict[key].sum()/value.sum()
            pissa_5iter_err_rate += pissa_5iter_error_r128_singular_value_dict[key].sum()/value.sum()
    print(loftq_1iter_err_rate/32, loftq_5iter_err_rate/32, pissa_1iter_err_rate/32, pissa_5iter_err_rate/32)