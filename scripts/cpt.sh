#!/usr/bin/env bash


cd LLaMA-Factory

pretrained_model=Gemma2-9B
outputs=checkpoints
dataset_name=GEMMAX2-28-CPT  # add a dataset description in dataset_info.json before training to use it


llamafactory-cli train \
    --model_name_or_path ${pretrained_model} \
    --stage pt \
    --do_train true \
    --seed 42 \
    --preprocessing_num_workers 64 \
    --finetuning_type full \
    --template empty \
    --flash_attn disabled \
    --dataset_dir data \
    --dataset ${dataset_name} \
    --cutoff_len 2048 \
    --learning_rate 2e-05 \
    --num_train_epochs 1 \
    --max_samples 100000000000 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 24 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.01 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --optim adamw_torch \
    --bf16 true \
    --logging_strategy steps \
    --logging_steps 5 \
    --save_steps 636 \
    --packing true \
    --plot_loss true \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen true \
    --output_dir ${outputs} \
    --deepspeed examples/deepspeed/ds_z2_config.json
