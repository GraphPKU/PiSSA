BASE_MODEL="meta-llama/Llama-2-7b-hf"
OUTPUT="output/full_finetune-llama-2-7b"

python pissa.py \
    --model_name_or_path $BASE_MODEL \
    --output_dir $OUTPUT \
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

python merge_adapter_to_base_model.py --base_mode $BASE_MODEL --adapter $OUTPUT/ft/ --output_path $OUTPUT
python inference/gsm8k_inference.py --model $OUTPUT
python inference/MATH_inference.py --model $OUTPUT