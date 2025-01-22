BASE_MODEL="meta-llama/Llama-2-7b-hf"
RES_MODEL="output/CLOVER-Llama-2-7b-h32-d128"
OUTPUT_PATH="output/metamath-CLOVER-Llama-2-h32-d128"
DATA_PATH="pissa-dataset"
export HF_ENDPOINT=https://hf-mirror.com

#huggingface-cli download --token hf_*** --resume-download $RES_MODEL --local-dir $RES_MODEL
if [ -e $RES_MODEL ]; then
    echo "Use pre-initialized residual model."
else
    echo "Perform CLOVER initialization by my self."
    python utils/init_clover.py --base_model_path $BASE_MODEL --output_dir $RES_MODEL --init_weights svd --target_modules q_proj k_proj v_proj o_proj --num_head 32 --head_dim 128
fi

# batch size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus = 128
deepspeed --master_port=16971 --include=localhost:0,1,2,3,4,5,6,7 train.py \
    --deepspeed configs/ds_config_zero2_no_offload.json \
    --model_name_or_path $RES_MODEL \
    --full_finetune False \
    --bf16 \
    --adapter_name_or_path "clover_init" \
    --data_path $DATA_PATH \
    --sub_task metamath:100000 \
    --dataset_split train \
    --dataset_field instruction output \
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 1 \
    --model_max_length 512 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \
    --report_to "tensorboard" \
    --merge True \

python utils/gen_vllm.py --model $OUTPUT_PATH --sub_task metamath --output_file $OUTPUT_PATH/metamath_response.jsonl
python utils/test_acc.py --input_file $OUTPUT_PATH/metamath_response.jsonl