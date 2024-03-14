# PiSA


# Install :

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

# PiSA Initialization ([code](https://github.com/fxmeng/peft/blob/a320ff7167816c963e846ce4091bbc860e5ff541/src/peft/tuners/lora/layer.py#L153))

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


# How to use PiSA to initialize a linear LoRA layer:
```
from peft.tuners.lora import Linear
import torch
import torch.nn as nn

x = torch.randn(16, 50, 1024)
layer = nn.Linear(1024, 4096, bias=False)
print(layer(x).sum())

# Vanilla LoRA (zero initialized)
lora_layer = Linear(layer, r=16, adapter_name='default', lora_alpha=16)
print(lora_layer(x).sum())

# PiSA-initialized LoRA (initialized with the principal singular values and vectors)
pisa_layer = Linear(layer, init_lora_weights='pisa_init', adapter_name='default', r=16, lora_alpha=16)
print(pisa_layer(x).sum())
```