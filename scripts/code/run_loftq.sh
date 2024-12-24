# LoftQ only provide model with rank=64, one can DIY a rank=128 version following:
# https://github.com/yxli2123/LoftQ/tree/main
RESIDUAL_MODEL="LoftQ/Meta-Llama-3-8B-4bit-64rank"
OUTPUT_PATH="output/LoftQ-Llama-3-8B-4bit-64rank"
DATA_PATH="meta-math/MetaMathQA"

# batch size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus = 128
deepspeed --master_port=16971 --include=localhost:0 train.py \
    --deepspeed configs/ds_config_zero2_no_offload.json \
    --model_name_or_path $RESIDUAL_MODEL \
    --full_finetune False \
    --bf16 \
    --bits 4 \
    --use_lora True \
    --adapter_name_or_path "loftq_init" \
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
