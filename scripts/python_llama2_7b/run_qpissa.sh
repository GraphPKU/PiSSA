BASE_MODEL="meta-llama/Llama-2-7b-hf"
RES_MODEL="output/QPiSSA-Llama-2-7b-4bit-r128-5iter"
OUTPUT_PATH="output/python-QPiSSA-Llama-2-7b-4bit-r128-5iter"
DATA_PATH="pissa-dataset"

if [ -e $RES_MODEL ]; then
    echo "Use pre-initialized residual model."
else
    echo "Perform QPiSSA initialization by my self."
    python utils/init_qpissa.py --base_model_dir $BASE_MODEL --output_path $RES_MODEL --rank 128 --iter 5 --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
fi

# batch size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus = 128
deepspeed --master_port=16971 --include=localhost:0,1,2,3,4,5,6,7 train.py \
    --deepspeed configs/ds_config_zero2_no_offload.json \
    --model_name_or_path $RES_MODEL \
    --full_finetune False \
    --bf16 \
    --bits 4 \
    --adapter_name_or_path "qpissa_init" \
    --data_path $DATA_PATH \
    --sub_task python \
    --dataset_split train \
    --dataset_field instruction output \
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

python utils/merge_adapter.py --base_model $RES_MODEL --adapter $OUTPUT_PATH/checkpoint-819/ --output_path $OUTPUT_PATH
python utils/gen_vllm.py --model $OUTPUT_PATH --sub_task python --output_file $OUTPUT_PATH/python_response.jsonl
python utils/code_process.py --path $OUTPUT_PATH/python_response.jsonl
evalplus.evaluate --dataset humaneval --samples $OUTPUT_PATH/humaneval.jsonl
evalplus.evaluate --dataset mbpp --samples $OUTPUT_PATH/mbpp.jsonl