# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import os
from peft import CloverConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

parser = argparse.ArgumentParser(description="Merge Adapter to Base Model")
parser.add_argument("--base_model_name_or_path", type=str, help="The name or path of the fp32/16 base model.")
parser.add_argument("--output_dir", type=str, default="clover_model")
parser.add_argument("--bits", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
parser.add_argument("--init_weights", type=str, default="eye", help="(`['eye', 'qr']`)")
parser.add_argument('--target_modules', nargs='+', help='', required=True)
script_args = parser.parse_args()
print(script_args)

model = AutoModelForCausalLM.from_pretrained(
    script_args.base_model_name_or_path,
    torch_dtype=(
        torch.float16
        if script_args.bits == "fp16"
        else (torch.bfloat16 if script_args.bits == "bf16" else torch.float32)
    ),
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(script_args.base_model_name_or_path)
clover_config = CloverConfig(
    init_clover_weights=script_args.init_weights,
    target_modules=script_args.target_modules,
    task_type="CAUSAL_LM",
)
peft_model = get_peft_model(model, clover_config)

# Save CLOVER modules:
peft_model.peft_config["default"].init_clover_weights = "eye"
peft_model.save_pretrained(os.path.join(script_args.output_dir, "clover_init"))
# Save residual model:
peft_model = peft_model.unload()
peft_model.save_pretrained(script_args.output_dir)
# Save the tokenizer:
tokenizer.save_pretrained(script_args.output_dir)