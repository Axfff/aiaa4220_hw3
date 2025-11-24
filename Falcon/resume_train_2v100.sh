#!/bin/bash
# Resume training script for 2x V100 GPUs with DD-PPO
# 恢复训练脚本 - 从最新checkpoint继续训练

set -e

echo "=========================================="
echo "Falcon Training Resume on 2x V100 GPUs"
echo "=========================================="
echo ""

# Check GPU availability
echo "Checking available GPUs..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# Check disk space (prevent another crash)
echo "Checking disk space..."
df -h / | grep -v Filesystem
DISK_USAGE=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
if [ $DISK_USAGE -gt 95 ]; then
    echo "⚠️  WARNING: Disk usage is ${DISK_USAGE}% - may cause training crash!"
    echo "Please clean up disk space before resuming training."
    exit 1
fi
echo "✓ Disk space OK: ${DISK_USAGE}% used"
echo ""

# Set environment variables for optimal performance
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

# Set random seeds for reproducibility
export PYTHONHASHSEED=42
export HABITAT_SEED=42
export CUDA_LAUNCH_BLOCKING=1

# Enable NCCL optimizations for V100
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo

# PyTorch distributed settings
export MASTER_ADDR=localhost
export MASTER_PORT=29500

echo "Environment variables set for resumption..."
echo ""

# Check for resume state - container path mapping
CHECKPOINT_DIR="/app/Falcon/evaluation/falcon/hm3d_2v100_optimized/checkpoints"
RESUME_STATE="$CHECKPOINT_DIR/.habitat-resume-state.pth"

if [ -f "$RESUME_STATE" ]; then
    echo "✓ Found resume state: $RESUME_STATE"
    ls -lah "$RESUME_STATE"
    echo ""
else
    echo "❌ No resume state found at $RESUME_STATE"
    echo "Available checkpoints:"
    ls -la "$CHECKPOINT_DIR"
    exit 1
fi

# Create timestamped log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="/app/Falcon/training_resume_${TIMESTAMP}.log"

echo "🔄 Resuming training from checkpoint..."
echo "📝 Full output logged to: $LOG_FILE"
echo "   Use 'tail -f $LOG_FILE' to monitor progress"
echo ""
# Pre-download ResNet50 ImageNet weights to avoid duplicate downloads
echo "Pre-downloading ResNet50 ImageNet weights..."
python3 -c "
import torch
import torchvision.models as models
print('Downloading ResNet50 pretrained weights...')
_ = models.resnet50(pretrained=True)
print('✓ ResNet50 weights cached successfully')
"
echo ""

echo "Starting distributed training with 2 GPUs..."
echo "Config: social_nav_v2/falcon_hm3d_train_2v100_optimized.yaml (OPTIMIZED)"
echo ""
echo "Optimizations:"
echo "  ✓ num_environments: 20 (RAM released, max performance)"
echo "  ✓ num_mini_batch: 10"
echo "  ✓ total_num_steps: 3,276,800 (1600 updates - 1.5x increase)"
echo "  ✓ Dynamic social penalty system enabled"
echo "  ✓ Warmup+cosine LR schedule with eta_min"
echo "  ✓ Best checkpoint: Save only when success rate improves (>1%)"
echo ""

# Launch DD-PPO training with 2 processes (one per GPU) - SAME AS train_2v100_fixed.sh
# Using torchrun (recommended) instead of deprecated torch.distributed.launch
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=2 \
    --master_port=$MASTER_PORT \
    -m habitat_baselines.run \
    --config-name=social_nav_v2/falcon_hm3d_train_2v100_optimized.yaml \
    habitat_baselines.load_resume_state_config=True 2>&1 | tee "$LOG_FILE"

echo ""
echo "=========================================="
echo "Training completed or interrupted!"
echo "Full log saved to: $LOG_FILE"
echo "=========================================="