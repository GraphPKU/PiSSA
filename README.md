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
    u,s,v = self.base_layer.weight.data.svd()
    ur = u[:,:self.r[adapter_name]]
    sr = torch.sqrt(s[:self.r[adapter_name]])
    vr = v[:,:self.r[adapter_name]].t()
    self.lora_A[adapter_name].weight.data = torch.mm(torch.diag(sr), vr)
    self.lora_B[adapter_name].weight.data = torch.mm(ur, torch.diag(sr))
    self.base_layer.weight.data = self.base_layer.weight.data - torch.mm(self.lora_B[adapter_name].weight.data, self.lora_A[adapter_name].weight.data)
```


## How to use PiSA to initialize a linear LoRA layer :
```
from peft.tuners.lora import Linear
import torch
import torch.nn as nn

x = torch.randn(16, 50, 1024)
layer = nn.Linear(1024, 4096, bias=False)
print(layer(x).sum())

# Vanilla LoRA (zero initialized)
lora_layer = Linear(layer, r=16, lora_alpha=16, adapter_name='default')
print(lora_layer(x).sum())

# PiSA-initialized LoRA (initialized with the principal singular values and vectors)
pisa_layer = Linear(layer, r=16, lora_alpha=16, adapter_name='default', init_lora_weights='pisa_init')
print(pisa_layer(x).sum())
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

python train.py \
    --model_name_or_path ${local or remote model} \
    --data_path data/MetaMathQA-395K.json \
    --output_dir output/model-metamath \
    --init_lora_weights ${siqa|lora|fp} \
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
    --tf32 True \
    --report_to ${none|tensorboard|wandb}

# Evaluation command:

python inference/gsm8k_inference.py --data_file data/gsm8k_test.jsonl --model output/model-metamath --batch_size 60 --tensor_parallel_size 1
python inference/MATH_inference.py --data_file data/MATH_test.jsonl --model output/model-metamath --batch_size 60 --tensor_parallel_size 1
```