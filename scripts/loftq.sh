RESIDUAL_MODEL="LoftQ/Llama-2-7b-hf-4bit-64rank"
OUTPUT="output/loftq-llama-2-7b-r64"

python qpissa.py \
    --model_name_or_path $RESIDUAL_MODEL \
    --output_dir $OUTPUT \
    --adapter_name_or_path loftq_init \
    --init_lora_weights loftq \
    --data_path meta-math/MetaMathQA \
    --dataset_split "train[:100000]"\
    --dataset_field query response \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --bf16 True \
    --tf32 True \
    --report_to none

# In gsm8k_inference.py, `from vllm import LLM, SamplingParams` does not support the nf4 model.
# To compare the performance of PiSSA and LoftQ, please create a bf16 LoftQ model using the instructions in LoftQ/quantize_save.py.
