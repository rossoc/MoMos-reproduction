#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define your parameter sets (row,col,capacity)
BEST_PARAMS=(
    "4,32,0.25"
)

# Process each parameter set
for entry in "${BEST_PARAMS[@]}"; do
    # Cleanly split the comma-separated values without altering global IFS
    IFS=',' read -r row col cap <<< "$entry"

    echo "================================================"
    echo " Running: rows=$row, cols=$col, capacity=$cap"
    echo "================================================"

    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    uv run python src/train.py \
        seed=42 \
        epochs=400 \
        accelerator=cuda \
        periodic_checkpoint=true \
        dataset.name=cifar10 \
        model=tinyvit \
        fold=1 \
        prefix=tinyVit_cifar10 \
        \
        wandb.enabled=true \
        wandb.project="momos-collapse" \
        wandb.name="momos_2d_c${col}_r${row}_cap${cap}_s42" \
        \
        "metrics=[sparsity,l2,gzip,bz2,lzma,bdm,qbdm]" \
        quantization=hierarchical_momos2d \
        quantization.switch_fraction=0 \
        \
        quantization.primary.cols=1 \
        quantization.primary.rows=2 \
        quantization.primary.capacity=0.001 \
        \
        quantization.secondary.rows="$row" \
        quantization.secondary.cols="$col" \
        quantization.secondary.capacity="$cap"

done
