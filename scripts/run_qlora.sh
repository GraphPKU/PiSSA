BASE_MODEL="meta-llama/Meta-Llama-3-8B"
OUTPUT_PATH="output/QLoRA-Llama-3-8B-4bit-r128"
DATA_PATH="meta-math/MetaMathQA"

# batch size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus = 128
deepspeed --master_port=16971 --include=localhost:1 pissa.py \
    --model_name_or_path $BASE_MODEL \
    --bf16 \
    --bits 4 \
    --use_lora True \
    --init_lora_weights True \
    --target_modules "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj" \
    --lora_rank 128 \
    --lora_alpha 128 \
    --lora_dropout 0 \
    --data_path $DATA_PATH \
    --dataset_field query response \
    --dataset_split "train[:100000]"\
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 1 \
    --model_max_length 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 128 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 100 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \
    --report_to "tensorboard" \
