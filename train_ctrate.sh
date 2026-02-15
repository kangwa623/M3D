#!/bin/bash

# ============================================================================
# M3D Training Script for CT-RATE Dataset
# ============================================================================
# This script trains M3D model on CT-RATE dataset using 3 GPUs (0, 1, 2)
# ============================================================================

# GPU Configuration - Use GPUs 0, 1, 2 (avoid GPU 3 which is busy)
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Set Python path
export PYTHONPATH=/nfs/usrhome2/africanstu/kangwa/m3d/M3D:$PYTHONPATH

# Navigate to project directory
cd /nfs/usrhome2/africanstu/kangwa/m3d/M3D

# Activate conda environment
source /home/africanstu/miniconda3/etc/profile.d/conda.sh
conda activate /nfs/usrhome2/africanstu/miniconda3/envs/m3d

# Check if transformers is installed
echo "Checking dependencies..."
python -c "import transformers" 2>/dev/null || {
    echo "Installing transformers..."
    pip install transformers==4.42.3
}

# Create output directory if it doesn't exist
mkdir -p ./LaMed/output/LaMed-Phi3-4B-pretrain

# Verify GPU availability
echo "=========================================="
echo "GPU Configuration"
echo "=========================================="
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
python -c "import torch; print(f'PyTorch sees {torch.cuda.device_count()} GPU(s)')"
nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv,noheader | grep -E "^[0-2],"

# Clear any cached accelerate config that might cause issues
echo ""
echo "Clearing cached accelerate config..."
rm -f ~/.cache/huggingface/accelerate/default_config.yaml 2>/dev/null || true

echo ""
echo "=========================================="
echo "Starting Training"
echo "=========================================="

# Run training with accelerate
# CRITICAL: Use --config_file to override any cached config
# CRITICAL: Explicitly set num_processes=3 to match 3 visible GPUs
accelerate launch \
    --config_file deepspeed.yaml \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision bf16 \
    LaMed/src/train/train.py \
    --version v0 \
    --model_name_or_path microsoft/Phi-3-mini-4k-instruct \
    --model_type lamed_phi3 \
    --vision_tower vit3d \
    --pretrain_vision_model ./LaMed/pretrained_model/M3D-CLIP/pretrained_ViT.bin \
    --tune_mm_mlp_adapter True \
    --bf16 True \
    --output_dir ./LaMed/output/LaMed-Phi3-4B-pretrain \
    --data_root /nfs/usrhome2/africanstu/kangwa/m3d/M3D/ctrate_volumes/m3d_npy \
    --cap_data_path ./Data/ctrate_dataset.json \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "steps" \
    --eval_accumulation_steps 1 \
    --eval_steps 0.04 \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 2 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --gradient_checkpointing False \
    --dataloader_pin_memory True \
    --dataloader_num_workers 8 \
    --report_to tensorboard \
    2>&1 | tee ./LaMed/output/LaMed-Phi3-4B-pretrain/training.log

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="