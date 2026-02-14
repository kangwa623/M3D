cd /nfs/usrhome2/africanstu/kangwa/m3d/M3D

# Activate environment
source /home/africanstu/miniconda3/etc/profile.d/conda.sh
conda activate /nfs/usrhome2/africanstu/miniconda3/envs/m3d

# Install required packages
echo "Installing accelerate and deepspeed..."
pip install accelerate==0.32.1 deepspeed==0.14.4

# Verify installation
echo ""
echo "=== Verifying installation ==="
python -c "import accelerate; print(f'Accelerate version: {accelerate.__version__}')" || echo "Accelerate not found"
python -c "import deepspeed; print(f'DeepSpeed version: {deepspeed.__version__}')" || echo "DeepSpeed not found"

# Check if accelerate command works
which accelerate || echo "accelerate command not in PATH"

# Update train_ctrate.sh to use python -m if needed
cat > train_ctrate.sh << 'SCRIPT_EOF'
#!/bin/bash

# Use GPUs 0, 1, 2 (avoid GPU 3 which is busy)
export CUDA_VISIBLE_DEVICES=0,1,2
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Navigate to project
cd /nfs/usrhome2/africanstu/kangwa/m3d/M3D

# Activate environment
source /home/africanstu/miniconda3/etc/profile.d/conda.sh
conda activate /nfs/usrhome2/africanstu/miniconda3/envs/m3d

# Try accelerate command first, fallback to python -m
if command -v accelerate &> /dev/null; then
    ACCELERATE_CMD="accelerate launch"
else
    ACCELERATE_CMD="python -m accelerate.commands.launch"
fi

# Run training with explicit num_processes to ensure 3 GPUs
$ACCELERATE_CMD \
    --num_processes 3 \
    --num_machines 1 \
    --mixed_precision bf16 \
    --config_file deepspeed.yaml \
    LaMed/src/train/train.py \
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
SCRIPT_EOF

chmod +x train_ctrate.sh

echo ""
echo "=== Script updated ==="
echo "Now try: bash train_ctrate.sh"