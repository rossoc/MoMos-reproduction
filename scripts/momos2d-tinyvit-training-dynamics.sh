#!/bin/bash
set -e

export PATH="$HOME/.local/bin:$PATH"

# Hierarchical MoMos2D params (from momos2d-training-dynamics.sh)
ROW=4
COL=32
CAP=0.25

echo "------------------------------------------------"
echo "Running V-Fold MoMos (TinyViT, no k-fold): rows=$ROW, cols=$COL, cap=$CAP"
echo "------------------------------------------------"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python src/train.py \
    epochs=400 \
    accelerator=cuda \
    periodic_checkpoint=true \
    dataset.name=cifar10 \
    prefix=tinyVit_cifar10 \
    model=tinyvit \
    seed=42 \
    wandb.enabled=true \
    wandb.name="momos_2d_c${COL}_r${ROW}_cap${CAP}_s42" \
    wandb.project="momos-collapse" \
    "metrics=[sparsity,l2,gzip,bz2,lzma,bdm,qbdm]" \
    quantization=hierarchical_momos2d \
    quantization.switch_fraction=0 \
    quantization.primary.cols=1 \
    quantization.primary.rows=2 \
    quantization.primary.capacity=0.001 \
    quantization.secondary.rows=$ROW \
    quantization.secondary.cols=$COL \
    quantization.secondary.capacity=$CAP
