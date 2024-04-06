# **P**r**i**ncipal **S**ingular values and **S**ingular vectors **A**daptation

As the parameters of large language models (LLMs) expand, the computational cost of fine-tuning the entire model becomes prohibitive. To address this challenge, we introduce a parameter-efficient fine-tuning (PEFT) method, \textbf{P}r\textbf{i}ncipal \textbf{S}ingular values and \textbf{S}ingular vectors \textbf{A}daptation (PiSSA), which optimizes a significantly reduced parameter space while achieving or surpassing the performance of full-parameter fine-tuning. PiSSA is inspired by Intrinsic SAID, which suggests that pre-trained, over-parametrized models inhabit a space of low intrinsic dimension. Consequently, PiSSA represents a matrix $W\in\mathbb{R}^{m\times n}$ within the model by the product of two trainable matrices $A \in \mathbb{R}^{m\times r}$ and $B \in \mathbb{R}^{r\times n}$, where $r \ll \min(m, n)$, plus a residual matrix $W^{res}\in\mathbb{R}^{m\times n}$ for error correction. Singular value decomposition (SVD) is employed to factorize $W$, and the principal singular values and vectors of $W$ are utilized to initialize $A$ and $B$. The residual singular values and vectors initialize the residual matrix $W^{res}$, which keeps frozen during fine-tuning. Notably, PiSSA shares the same architecture with Low-Rank Adaptation (LoRA), which hypothesizes that changes in model parameters $\Delta W$ form a low-rank matrix. However, LoRA approximates $\Delta W$ through the product of two matrices, $A$, initialized with Gaussian noise, and $B$, initialized with zeros, while PiSSA initializes $A$ and $B$ with principal singular values and singular vectors of the original matrix $W$. Given that the principal singular values and vectors capture the essence of a low-rank matrix, PiSSA can better approximate the outcomes of full-parameter fine-tuning at the beginning by changing the essential parts while freezing the ``noisy'' parts. In comparison, LoRA freezes the original matrix and updates the ``noise''. This distinction enables PiSSA to convergence much faster than LoRA and also achieve better performance in the end. On five common benchmarks, PiSSA outperforms LoRA on all of them using exactly the same setups except for a different initialization. On GSM8K, Mistral-7B fine-tuned with PiSSA achieves an accuracy of 72.86\%, outperforming LoRA's 67.7\% by 5.16\%.
Due to the same architecture, PiSSA inherits many of LoRA's advantages, such as parameter efficiency and compatibility with quantization. Leveraging a fast SVD method, the initialization of PiSSA takes only a few seconds, inducing negligible cost of switching LoRA to PiSSA.

![PiSSA](./assets/full-lora-pissa.png)
![GSM8K](./assets/gsm8k.png)


## Quickstart :

<details open>
<summary>1. Install PiSSA via pip:</summary>

    pip install git+https://github.com/fxmeng/peft.git
</details>


<details>
<summary>[Optional] Installation from Source Code:</summary>

    git clone https://github.com/fxmeng/peft.git
    cd peft

    # To modify the implementation, you can edit the file by:
    # vim src/peft/tuners/lora/layer.py # L154-L186
    # and adjust the pissa_init method as shown below:
    # def pissa_init(self, adapter_name):
    #     assert self.scaling[adapter_name] == 1
    #     U, S, Vh = torch.linalg.svd(self.base_layer.weight.data, full_matrices=False)
    #     Ur = U[:,:self.r[adapter_name]]
    #     Sr = S[:self.r[adapter_name]]
    #     Vhr = Vh[:self.r[adapter_name]]
    #     lora_A = torch.diag(torch.sqrt(Sr)) @ Vhr
    #     lora_B = Ur @ torch.diag(torch.sqrt(Sr))
    #     self.lora_A[adapter_name].weight.data = lora_A
    #     self.lora_B[adapter_name].weight.data = lora_B
    #     self.base_layer.weight.data = self.base_layer.weight.data - lora_B @ lora_A

    pip install -e .
</details>

<details open>
<summary>2. Initializing PiSSA and the residual model with SVD:</summary>

    # Download the standard llama-2-7b model from huggingface:

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b', device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b')
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Inject PiSSA to the base model:

    from peft import LoraConfig, get_peft_model
    peft_config = LoraConfig(
        r = 16,
        lora_alpha = 16, # lora_alpha should match r to maintain scaling = 1
        lora_dropout = 0,
        init_lora_weights='pissa', # PiSSA initialization
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
</details>


<details>
<summary>[Optional] Fast SVD Initialization for PiSSA:</summary>

    # Download the llama-2-7b model from huggingface:

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b', device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b')
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Configure PiSSA with Fast SVD:

    from peft import LoraConfig, get_peft_model
    peft_config = LoraConfig(
        r = 16,
        lora_alpha = 16,
        lora_dropout = 0,
        init_lora_weights='pissa_niter_4', # Fast initialization with "_niter_xx"
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
</details>



<details>
<summary>[Optional] Fast SVD and 4-bit Quantization for PiSSA</summary>

    # Download and load the llama-2-7b model in 4-bit format:

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b', quantization_config=config)
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b')
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # wrapping the model with PiSSA:

    from peft import LoraConfig, get_peft_model
    peft_config = LoraConfig(
        r = 16,
        lora_alpha = 16,
        lora_dropout = 0,
        init_lora_weights='pissa_niter_4', # Accelerated initialization with "_niter_xx"
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model = prepare_model_for_kbit_training(model)
    model.print_trainable_parameters()
</details>


<details open>
<summary>3. Finetune PiSSA on Alpaca Dataset:</summary>

    from trl import SFTTrainer
    from datasets import load_dataset
    dataset = load_dataset("fxmeng/alpaca_in_mixtral_format", split="train")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=1024,
        tokenizer=tokenizer
    )
    trainer.train()
</details>


<details open>
<summary>4. save and sharing your fine-tuned PiSSA to Hugging Face:</summary>

    model.save_pretrained('pissa-r16-llama-2-7b-alpaca') # Save locally
    model.push_to_hub('username/pissa-r16-llama-2-7b-alpaca') # # Push to Hugging Face
</details>

<details>
<summary>[Optional] Convert PiSSA to LoRA for sharing to Hugging Face</summary>

    ### It's essential to save initial PiSSA parameters for conversion to LoRA. ###

    model.save_pretrained('pissa-r16-llama-2-7b-alpaca-init')

    ### trainer.train()... ###

    ### Upon completion, save final PiSSA parameters ###
    model.save_pretrained('pissa-r16-llama-2-7b-alpaca-finetuned')

    import os
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    import json

    def pissa_to_lora(init_path, finetuned_path, output_path, device='cpu', tensors_name="adapter_model.safetensors", config_name="adapter_config.json"):
        tensors_init = {}
        with safe_open(os.path.join(init_path, tensors_name), framework="pt", device=device) as f:
            for k in f.keys():
                tensors_init[k] = f.get_tensor(k)
                
        tensors_finetune = {}
        with safe_open(os.path.join(finetuned_path, tensors_name), framework="pt", device=device) as f:
            for k in f.keys():
                tensors_finetune[k] = f.get_tensor(k)
                
        tensors_delta_w = {}
        for name in tensors_init.keys():
            tensors_delta_w[name] = torch.cat([tensors_finetune[name], -tensors_init[name]], dim=0 if 'lora_A' in name else 1)

        if not os.path.exists(output_path):
            os.mkdir(output_path)
        save_file(tensors_delta_w, os.path.join(output_path, tensors_name))
        
        with open(os.path.join(init_path, config_name))as f:
            adapter_config = json.load(f)
        adapter_config['init_lora_weights']=True
        adapter_config['r']*=2
        adapter_config['lora_alpha']*=2
        with open(os.path.join(output_path, config_name),'w')as f:
            json.dump(adapter_config, f)
    
    ### The different of the PiSSA parameters before and after the training corresponding to delta W in LoRA. ###
    pissa_to_lora('pissa-r16-llama-2-7b-alpaca-init', 'pissa-r16-llama-2-7b-alpaca-finetuned', "pissa-r16-llama-2-7b-alpaca-delta_w", device='cpu')

    ### Finally, create a new Hugging Face repository and upload the converted files... ###
</details>


<details open>
<summary>5. Loading PiSSA from Local or Hugging Face:</summary>

    from peft import PeftModel
    base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b", device_map='auto')
    model = PeftModel.from_pretrained(base_model, 'username/pissa-r16-llama-2-7b-alpaca')
</details>


<details>
<summary>[Optional] Loading PiSSA-Converted LoRA from Local or Hugging Face:</summary>

    from peft import PeftModel
    base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b", device_map='auto')
    model = PeftModel.from_pretrained(base_model, 'username/pissa-r16-llama-2-7b-alpaca-delta_w')
</details>


## Citation
```
@misc{meng2024pissa,
      title={PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models}, 
      author={Fanxu Meng and Zhaohui Wang and Muhan Zhang},
      year={2024},
      eprint={2404.02948},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```