#!/usr/bin/env bash


cd LlamaFactory

pretrained_model=MiMT-46-12B-Pretrain
outputs=checkpoints
dataset_name=MiMT-46-SFT  # Add a dataset description to dataset_info.json before training to enable its use.


llamafactory-cli train \
    --model_name_or_path ${pretrained_model} \
    --stage sft \
    --do_train true \
    --seed 42 \
    --preprocessing_num_workers 64 \
    --finetuning_type full \
    --template empty \
    --flash_attn fa2 \
    --dataset_dir data \
    --dataset ${dataset_name} \
    --cutoff_len 4096 \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --max_samples 100000000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type inverse_sqrt \
    --warmup_ratio 0.01 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --optim adamw_torch \
    --bf16 true \
    --prediction_loss_only true \
    --logging_strategy steps \
    --logging_steps 5 \
    --save_strategy epoch \
    --save_only_model true \
    --save_total_limit 1 \
    --plot_loss true \
    --ddp_timeout 180000000 \
    --overwrite_cache true \
    --overwrite_output_dir true \
    --output_dir ${outputs} \
    --deepspeed examples/deepspeed/ds_z2_config.json
