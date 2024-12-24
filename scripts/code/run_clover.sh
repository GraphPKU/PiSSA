MODEL_PATH="meta-llama/Llama-2-7b-hf"
OUTPUT_PATH="output/CLOVER-Llama-2-7b-hf"
DATA_PATH="fxmeng/commonsense-170k"

python init_clover.py --base_model_name_or_path $MODEL_PATH --output_dir $OUTPUT_PATH/INIT --init_weights qr --target_modules q_proj k_proj v_proj
# batch size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus = 16
deepspeed --master_port=16971 --include=localhost:0,1,2,3,4,5,6,7 train.py \
    --deepspeed configs/ds_config_zero2_no_offload.json \
    --model_name_or_path $OUTPUT_PATH/INIT \
    --adapter_name_or_path "clover_init" \
    --full_finetune False \
    --bf16 \
    --data_path $DATA_PATH \
    --dataset_field instruction output \
    --dataset_split "train"\
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 3 \
    --model_max_length 512 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 100000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_steps 100 \
    --logging_steps 1 \
    --lr_scheduler_type "linear" \
    --report_to "tensorboard" \
    --merge True \


python inference/gen_vllm.py --model $OUTPUT_PATH --data_path inference/data/eval_boolq/ --output_file boolq_response.jsonl --max_tokens 32
python inference/gen_vllm.py --model $OUTPUT_PATH --data_path inference/data/eval_piqa/ --output_file piqa_response.jsonl --max_tokens 32
python inference/gen_vllm.py --model $OUTPUT_PATH --data_path inference/data/eval_social_interaction_qa/ --output_file social_interaction_qa_response.jsonl --max_tokens 32
python inference/gen_vllm.py --model $OUTPUT_PATH --data_path inference/data/eval_hellaswag/ --output_file hellaswag_response.jsonl --max_tokens 32
python inference/gen_vllm.py --model $OUTPUT_PATH --data_path inference/data/eval_winogrande/ --output_file winogrande_response.jsonl --max_tokens 32
python inference/gen_vllm.py --model $OUTPUT_PATH --data_path inference/data/eval_arc_easy/ --output_file arc_easy_response.jsonl --max_tokens 32
python inference/gen_vllm.py --model $OUTPUT_PATH --data_path inference/data/eval_arc_challenge/ --output_file arc_challenge_response.jsonl --max_tokens 32
python inference/gen_vllm.py --model $OUTPUT_PATH --data_path inference/data/eval_openbookqa/ --output_file openbookqa_response.jsonl --max_tokens 32
python inference/acc_commonsense.py --input_file $OUTPUT_PATH/boolq_response.jsonl --dataset boolq
python inference/acc_commonsense.py --input_file $OUTPUT_PATH/piqa_response.jsonl --dataset piqa
python inference/acc_commonsense.py --input_file $OUTPUT_PATH/social_interaction_qa_response.jsonl --dataset social_interaction_qa
python inference/acc_commonsense.py --input_file $OUTPUT_PATH/hellaswag_response.jsonl --dataset hellaswag
python inference/acc_commonsense.py --input_file $OUTPUT_PATH/winogrande_response.jsonl --dataset winogrande
python inference/acc_commonsense.py --input_file $OUTPUT_PATH/arc_easy_response.jsonl --dataset arc_easy
python inference/acc_commonsense.py --input_file $OUTPUT_PATH/arc_challenge_response.jsonl --dataset arc_challenge
python inference/acc_commonsense.py --input_file $OUTPUT_PATH/openbookqa_response.jsonl --dataset openbookqa