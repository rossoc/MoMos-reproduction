#!/bin/bash
set -e

COLS=(2)
ROWS=(2)
CAPACITIES=(0.001)
SEEDS=(42)

# Iterate through all combinations
for col in "${COLS[@]}"; do
    for row in "${ROWS[@]}"; do
        
        # Skip the case where it's a 1x1 grid
        if [[ $col -eq 1 && $row -eq 1 ]]; then
            echo "Skipping 1x1 configuration..."
            continue
        fi

        if [[ $((col * row)) -gt 8 ]]; then
            echo "Skipping configuration (product > 8)..."
            continue
        fi

        for cap in "${CAPACITIES[@]}"; do
            for seed in "${SEEDS[@]}"; do

                echo "------------------------------------------------"
                echo "Running: cols=$col, rows=$row, cap=$cap, seed=$seed"
                echo "------------------------------------------------"

                uv run python src/train.py \
                    epochs=100 \
                    dataset.name=cifar10 \
                    wandb.enabled=true \
                    wandb.project=momos2d-remake \
                    wandb.name="momos_2d_c${col}_r${row}_cap${cap}_s${seed}" \
                    "metrics=[sparsity,l2,gzip,bz2,lzma,bdm]" \
                    quantization.enabled=true \
                    quantization.method=momos2d \
                    quantization.cols=$col \
                    quantization.rows=$row \
                    quantization.capacity=$cap \
                    quantization.force_zero=true \
                    quantization.q=32 \
                    patience=10 \
                    seed=$seed
            done
        done
    done
done
