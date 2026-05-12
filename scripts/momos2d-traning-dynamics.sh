#!/bin/bash

# Define the best_params list: "row,col,capacity"
BEST_PARAMS=(
    "2,1,0.001"
    "8,1,0.001"
    "1,8,0.001"
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
        batch_size=256 \
        accelerator=cuda \
        periodic_checkpoint=true \
        dataset.name=cifar10 \
        wandb.enabled=true \
        wandb.name="momos_2d_c${col}_r${row}_cap${cap}_s42" \
        wandb.project="momos-collapse" \
        "metrics=[sparsity,l2,gzip,bz2,lzma,bdm]" \
        quantization.enabled=true \
        quantization.method=momos2d \
        quantization.cols=$col \
        quantization.rows=$row \
        quantization.capacity=$cap \
        quantization.force_zero=true \
        quantization.q=32 \
        seed=42

done
