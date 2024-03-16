# PiSA


## Install :

```
conda create -n pisa python=3.8 -y
conda activate pisa
pip install -U vllm
pip install -U xformers
pip install -U torch
pip install -U accelerate
pip install datasets
pip install transformers
pip install jsonlines
pip install Fraction
pip install openai
pip install human_eval
pip install git+https://github.com/fxmeng/peft.git
```

## PiSA Initialization ([code](https://github.com/fxmeng/peft/blob/a320ff7167816c963e846ce4091bbc860e5ff541/src/peft/tuners/lora/layer.py#L153)) :

```
def pisa_init(self, adapter_name):
    assert self.scaling[adapter_name] == 1
    U, S, Vh = torch.linalg.svd(self.base_layer.weight.data, full_matrices=False)
    Ur = U[:,:self.r[adapter_name]]
    Sr = S[:self.r[adapter_name]]
    Vhr = Vh[:self.r[adapter_name]]
    lora_A = torch.diag(torch.sqrt(Sr)) @ Vhr
    lora_B = Ur @ torch.diag(torch.sqrt(Sr))
    self.lora_A[adapter_name].weight.data = lora_A
    self.lora_B[adapter_name].weight.data = lora_B
    self.base_layer.weight.data = self.base_layer.weight.data - lora_B @ lora_A
```


## How to use PiSA to initialize a linear LoRA layer :
```
from peft.tuners.lora import Linear
import torch
import torch.nn as nn

x = torch.randn(16, 50, 1024)
layer = nn.Linear(1024, 4096, bias=False)
print(layer(x).sum())

# Vanilla LoRA (initializes with Kaiming-uniform/gaussian for weight A and zeros for weight B)
lora_layer = Linear(layer, r=16, lora_alpha=16, adapter_name='default')
print(lora_layer(x).sum())

# PiSA-initialized LoRA (initialized with the principal singular values and vectors)
pisa_layer = Linear(layer, r=16, lora_alpha=16, adapter_name='default', init_lora_weights='pisa_init')
print(pisa_layer(x).sum())
```

## How to use PiSA to initialize a model (taking about 2 minutes for 4090) :
```
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map='auto')
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
lora_config = LoraConfig(r=64, lora_alpha=64, init_lora_weights='pisa_init')
pisa_model = get_peft_model(base_model, lora_config)
```

## Save PiSA initialized LoRA layers and the residual model (W-USV) for fast loading
```
# Save LoRA layers
pisa_model.peft_config['default'].init_lora_weights = True # it's unnecessary using pisa initialization during loading
pisa_model.save_pretrained('pisa/pisa_init')

# Saving the residual model
lora_config = LoraConfig(r=64, lora_alpha=64)
pisa_model.add_adapter(adapter_name='zero_init', peft_config=lora_config)
pisa_model.merge_and_unload(adapter_names=['zero_init'])
base_model = pisa_model.get_base_model()
base_model.save_pretrained('pisa')
tokenizer.save_pretrained('pisa')


# Fast loading
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained("pisa", device_map='auto')
model = PeftModel.from_pretrained(base_model, 'pisa/pisa_init')
```

## 4-bit training
```
# Require saving PiSA initialized LoRA layers and the residual model first, and then:

import torch
from transformers import BitsAndBytesConfig
config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Quantize the residual model
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('pisa', quantization_config=config)

# Add pisa-initialized lora layers
from peft import PeftModel
model = PeftModel.from_pretrained(model, 'pisa/pisa_init')

# Allowing kbit training
from peft import prepare_model_for_kbit_training
model = prepare_model_for_kbit_training(model)
```


## All the datasets are under the folder ./data :
```
# Training set - data/MetaMathQA-395K.json:
{
    "query": "Gracie and Joe are choosing numbers on the complex plane. Joe chooses the point $1+2i$. Gracie chooses $-1+i$. How far apart are Gracie and Joe's points?", 
    "response": "The distance between two points $(x_1,y_1)$ and $(x_2,y_2)$ in the complex plane is given by the formula $\\sqrt{(x_2-x_1)^2+(y_2-y_1)^2}$.\nIn this case, Joe's point is $(1,2)$ and Gracie's point is $(-1,1)$.\nSo the distance between their points is $\\sqrt{((-1)-(1))^2+((1)-(2))^2}=\\sqrt{(-2)^2+(-1)^2}=\\sqrt{4+1}=\\sqrt{5}$.\nTherefore, Gracie and Joe's points are $\\boxed{\\sqrt{5}}$ units apart.\nThe answer is: \\sqrt{5}", "type": "MATH_AnsAug", "original_question": "Gracie and Joe are choosing numbers on the complex plane. Joe chooses the point $1+2i$. Gracie chooses $-1+i$. How far apart are Gracie and Joe's points?"
}

# Evaluation set data/gsm8k_test.jsonl:
{
    "question": "A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?", 
    "answer": "It takes 2/2=<<2/2=1>>1 bolt of white fiber\nSo the total amount of fabric is 2+1=<<2+1=3>>3 bolts of fabric\n#### 3"
}
...
# Evaluation set data/MATH_test.jsonl:
{
    "idx": "hendrycks_math_2", 
    "instruction": "Compute $\\arccos (-1).$  Express your answer in radians.", 
    "input": "", 
    "output": "Since $\\cos \\pi = -1,$ $\\arccos (-1) = \\boxed{\\pi}.$", 
    "type": "Precalculus"
}

```

## Running Scripts :
```
# Training command:
# --init_lora_weights = fp means finetune full parameters when --merge_and_save should be False.

python train.py \
    --model_name_or_path ${local or remote model} \
    --data_path data/MetaMathQA-395K.json \
    --output_dir ${output/model-metamath} \
    --init_lora_weights ${pisa|lora|fp} \
    --report_to ${none|tensorboard|wandb} \
    --query "query" \
    --response "response"\
    --merge_and_save True \
    --data_length 100000 \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True

# Evaluation command:

python inference/gsm8k_inference.py --data_file data/gsm8k_test.jsonl --model output/model-metamath --batch_size 60 --tensor_parallel_size 1
python inference/MATH_inference.py --data_file data/MATH_test.jsonl --model output/model-metamath --batch_size 60 --tensor_parallel_size 1
```