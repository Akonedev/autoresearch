#!/bin/bash
# Multi-GPU training launcher for autoresearch
# Usage: ./train_multigpu.sh [NUM_GPUS]

NUM_GPUS=${1:-2}  # Default to 2 GPUs

echo "======================================"
echo "  Autoresearch Multi-GPU Training"
echo "======================================"
echo ""
echo "Number of GPUs: $NUM_GPUS"
echo "Backend: $([ -n "$PYTORCH_CUDA" ] && echo "NCCL (CUDA)" || echo "Gloo (ROCm/CPU)")"
echo ""

# Check available GPUs
if command -v rocm-smi &> /dev/null; then
    echo "Available AMD GPUs:"
    rocm-smi --showproductname 2>/dev/null | grep -A1 "GPU\[" || echo "  (rocm-smi not available)"
    echo ""
fi

# Set environment variables for ROCm
export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm/hip

# Launch with torchrun (PyTorch's distributed launcher)
echo "Starting training with torchrun..."
echo ""

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="localhost" \
    --master_port=29500 \
    train.py

# Exit with the same code as torchrun
exit $?
