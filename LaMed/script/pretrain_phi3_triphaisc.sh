#!/bin/bash

# run "accelerate config" first!
export CUDA_DEVICE_ORDER=PCI_BUS_ID

PYTHONPATH=. CUDA_VISIBLE_DEVICES=6,7 accelerate launch --main_process_port 29600 --config_file deepspeed.yaml LaMed/src/train/train.py \
    --cap_data_path ./Data/m3d_triphasic_dataset.json \
    --version v0 \
    --model_name_or_path microsoft/Phi-3-mini-4k-instruct \
    --model_type lamed_phi3 \
    --vision_tower vit3d \
    --pretrain_vision_model /nfs/usrhome2/mkfmelbatel/M3D/M3D-CLIP/pretrained_ViT.bin \
    --tune_mm_mlp_adapter True \
    --bf16 True \
    --output_dir /nfs/usrhome2/mkfmelbatel/M3D/output/TriPhasicLaMed-Phi3-4B-pretrain-0000 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_accumulation_steps 1 \
    --eval_steps 0.04 \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 0.001 \
    --gradient_checkpointing False \
    --dataloader_pin_memory True\
    --dataloader_num_workers 8 \
    --report_to wandb
