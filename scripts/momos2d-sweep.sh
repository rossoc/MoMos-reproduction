#!/bin/bash
set -e

COLS=(1 2 4 8)
ROWS=(1 2 4 8)
CAPACITIES=(0.0001 0.001 0.01 0.1)
SEEDS=(42 777 50 78 291)

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
            # Loop using the indices of the SEEDS array (0 to 4)
            for i in "${!SEEDS[@]}"; do
                seed="${SEEDS[$i]}"
                fold=$i  # The fold is now directly linked to the seed index (0, 1, 2, 3, 4)

                echo "------------------------------------------------"
                echo "Running: cols=$col, rows=$row, cap=$cap, seed=$seed, fold=$fold"
                echo "------------------------------------------------"

                uv run python src/train.py \
                    epochs=100 \
                    dataset.name=cifar10 \
                    wandb.enabled=true \
                    wandb.project=momos2d-remake \
                    wandb.name="momos_2d_c${col}_r${row}_cap${cap}_s${seed}_f${fold}" \
                    "metrics=[sparsity,l2,gzip,bz2,lzma,bdm]" \
                    quantization.enabled=true \
                    quantization.method=momos2d \
                    quantization.cols=$col \
                    quantization.rows=$row \
                    quantization.capacity=$cap \
                    quantization.force_zero=true \
                    quantization.q=32 \
                    seed=$seed \
                    fold=$fold
            done
        done
    done
done
