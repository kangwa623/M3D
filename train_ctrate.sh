#!/bin/bash

# Use GPUs 0, 1, 2 (avoid GPU 3)
export CUDA_VISIBLE_DEVICES=0,2
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Navigate to project
 cd /nfs/usrhome2/africanstu/kangwa/m3d/M3D

# Activate environment
 conda activate /nfs/usrhome2/africanstu/miniconda3/envs/m3d

# Run training with accelerate (configure first: accelerate config)
accelerate launch --config_file deepspeed.yaml LaMed/src/train/train.py \
    --version v0 \
    --model_name_or_path microsoft/Phi-3-mini-4k-instruct \
    --model_type lamed_phi3 \
    --vision_tower vit3d \
    --pretrain_vision_model ./LaMed/pretrained_model/M3D-CLIP/pretrained_ViT.bin \
    --tune_mm_mlp_adapter True \
    --bf16 True \
    --output_dir ./LaMed/output/LaMed-Phi3-4B-pretrain \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --dataloader_num_workers 8