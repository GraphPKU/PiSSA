BASE_MODEL="meta-llama/Meta-Llama-3-8B"
OUTPUT_PATH="output/LoRA-Llama-3-8B-r128"
DATA_PATH="meta-math/MetaMathQA"

# batch size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus = 128
deepspeed --master_port=16971 --include=localhost:0,1,2,3,4,5,6,7 train.py \
    --deepspeed configs/ds_config_zero2_no_offload.json \
    --model_name_or_path $BASE_MODEL \
    --full_finetune False \
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
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 100 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \
    --report_to "tensorboard" \
    --merge True \

python inference/gen_vllm.py --model $OUTPUT_PATH/$lr --data_path inference/data/eval_gsm8k/ --output_file gsm8k_response.jsonl
python inference/gen_vllm.py --model $OUTPUT_PATH/$lr --data_path inference/data/eval_math/ --output_file math_response.jsonl
python inference/acc_gsm8k.py --input_file $OUTPUT_PATH//$lr/gsm8k_response.jsonl 
python inference/acc_math.py --input_file $OUTPUT_PATH/$lr/math_response.jsonl 
