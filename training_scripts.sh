# Training command:
python train.py \
    --model_name_or_path ${local or remote model} \
    --data_path data/MetaMathQA-395K.json \
    --output_dir ${output/model-metamath} \
    --init_lora_weights ${pissa|pissa_niter_4|lora|fp} \
    --report_to ${none|tensorboard|wandb} \
    --query "query" \
    --response "response"\
    --merge_and_save True \
    --data_length 100000 \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True

# Evaluation command:
python inference/gsm8k_inference.py --model output/model-metamath
python inference/MATH_inference.py --model output/model-metamath