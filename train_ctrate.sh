#!/bin/bash

# ============================================================================
# M3D Training Script for CT-RATE Dataset (Single GPU)
# ============================================================================
# Uses one GPU only. Paths are relative to the script directory.
# ============================================================================

set -e

# Project root: directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export M3D_ROOT="$SCRIPT_DIR"
export PYTHONPATH="$M3D_ROOT:$PYTHONPATH"

# ----------------------------------------------------------------------------
# GPU: use a single GPU (change to 1, 2, 3 if needed)
# ----------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# ----------------------------------------------------------------------------
# Optional: activate conda if available (comment out or adjust if not using conda)
# ----------------------------------------------------------------------------
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate m3d 2>/dev/null || true
elif [ -n "$CONDA_PREFIX" ]; then
    :  # already in conda
fi

# ----------------------------------------------------------------------------
# Output and data paths (relative to project root)
# ----------------------------------------------------------------------------
OUTPUT_DIR="$M3D_ROOT/LaMed/output/LaMed-Phi3-4B-pretrain"
DATA_ROOT="$M3D_ROOT/ctrate_volumes/m3d_npy"
CAP_JSON="$M3D_ROOT/Data/ctrate_dataset.json"
PRETRAIN_VIT="$M3D_ROOT/LaMed/pretrained_model/M3D-CLIP/pretrained_ViT.bin"

mkdir -p "$OUTPUT_DIR"

# ----------------------------------------------------------------------------
# Pre-flight checks
# ----------------------------------------------------------------------------
echo "=========================================="
echo "M3D CT-RATE Training (Single GPU)"
echo "=========================================="
echo "Project root: $M3D_ROOT"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Output dir:   $OUTPUT_DIR"
echo "Data root:    $DATA_ROOT"
echo "Caption JSON: $CAP_JSON"
echo "=========================================="

python -c "import torch; n=torch.cuda.device_count(); print(f'PyTorch sees {n} GPU(s)'); exit(0 if n>=1 else 1)" || {
    echo "Error: No GPU visible. Check CUDA_VISIBLE_DEVICES."
    exit 1
}

if [ ! -f "$CAP_JSON" ]; then
    echo "Warning: $CAP_JSON not found. Run create_ctrate_dataset_json.py after preprocess_v1.py"
    exit 1
fi

if [ ! -f "$PRETRAIN_VIT" ]; then
    echo "Warning: Pretrained ViT not found at $PRETRAIN_VIT"
    exit 1
fi

# Clear cached accelerate config so single-GPU settings take effect
rm -f ~/.cache/huggingface/accelerate/default_config.yaml 2>/dev/null || true

echo ""
echo "Starting training..."
echo ""

# ----------------------------------------------------------------------------
# Launch training: 1 process, 1 GPU
# ----------------------------------------------------------------------------
accelerate launch \
    --config_file "$M3D_ROOT/deepspeed.yaml" \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision bf16 \
    "$M3D_ROOT/LaMed/src/train/train.py" \
    --version v0 \
    --model_name_or_path microsoft/Phi-3-mini-4k-instruct \
    --model_type lamed_phi3 \
    --vision_tower vit3d \
    --pretrain_vision_model "$PRETRAIN_VIT" \
    --tune_mm_mlp_adapter True \
    --bf16 True \
    --output_dir "$OUTPUT_DIR" \
    --data_root "$DATA_ROOT" \
    --cap_data_path "$CAP_JSON" \
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
    --dataloader_num_workers 4 \
    --report_to tensorboard \
    2>&1 | tee "$OUTPUT_DIR/training.log"

echo ""
echo "=========================================="
echo "Training complete."
echo "=========================================="