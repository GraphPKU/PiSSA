RESIDUAL_MODEL="fxmeng/PiSSA-Llama-2-7B-r128"
OUTPUT="output/pissa-llama-2-7b-r128"

python pissa.py \
    --model_name_or_path $RESIDUAL_MODEL \
    --output_dir $OUTPUT \
    --adapter_name_or_path pissa_init \
    --init_lora_weights pissa \
    --data_path meta-math/MetaMathQA \
    --dataset_split "train[:100000]"\
    --dataset_field query response \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 128 \
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

CUDA_VISIBLE_DEVICES=0 python merge_adapter_to_base_model.py --base_mode $RESIDUAL_MODEL --adapter $OUTPUT/ft/ --output_path $OUTPUT
CUDA_VISIBLE_DEVICES=0 python inference/gsm8k_inference.py --model $OUTPUT
CUDA_VISIBLE_DEVICES=0 python inference/MATH_inference.py --model $OUTPUT