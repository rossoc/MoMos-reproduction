#!/bin/bash

# Define the best_params list: "row,col,capacity"
BEST_PARAMS=(
    "4,32,0.25"
)

# Iterate through the specific parameter sets
for entry in "${BEST_PARAMS[@]}"; do
    # Split the comma-separated string into variables
    IFS=',' read -r row col cap <<< "$entry"

    echo "------------------------------------------------"
    echo "Running: rows=$row, cols=$col, cap=$cap"
    echo "------------------------------------------------"

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python src/train.py \
        epochs=400 \
        accelerator=cuda \
        periodic_checkpoint=true \
        dataset.name=cifar10 \
        model=mlp \
        wandb.enabled=true \
        wandb.name="momos_fold_tinyvit" \
        wandb.project="momos-collapse" \
        "metrics=[sparsity,l2,gzip,bz2,lzma,bdm,qbdm]" \
        quantization=hierarchical_momos2d \
        quantization.switch_fraction=0 \
        quantization.primary.cols=1 \
        quantization.primary.rows=2 \
        quantization.primary.capacity=0.001 \
        quantization.secondary.rows=$row \
        quantization.secondary.cols=$col \
        quantization.secondary.capacity=$cap
        seed=42

done
