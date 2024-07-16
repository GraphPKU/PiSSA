RESIDUAL_MODEL="fxmeng/PiSSA-Llama-3-8B-4bit-r128-5iter"
OUTPUT_PATH="output/QPiSSA-Llama-3-8B-4bit-r128-5iter"
DATA_PATH="meta-math/MetaMathQA"

# batch size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus = 128
deepspeed --master_port=16971 --include=localhost:2 pissa.py \
    --model_name_or_path $RESIDUAL_MODEL \
    --bf16 \
    --bits 4 \
    --use_lora True \
    --adapter_name_or_path "pissa_init" \
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
