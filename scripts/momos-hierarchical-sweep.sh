#!/bin/bash
set -e

# Define the missing switch variable (adjust this value as needed)
SWITCH=0

COLS=(1 2 4 8 16 32 64 128)
ROWS=(1 2 4 8 16 32 64 128)
CAPACITIES=(0.0166 0.0083 0.00415 0.002075 0.0010375)
SEEDS=(42 777 50 78 291)

# Iterate through all combinations
for s_col in "${COLS[@]}"; do
    for s_row in "${ROWS[@]}"; do
        
        # Skip the case where it's a 1x1 grid
        if [[ $s_col -eq 1 && $s_row -eq 1 ]]; then
            echo "Skipping 1x1 configuration..."
            continue
        fi

        if [[ $((s_col * s_row)) -lt 64 ]]; then
            echo "Skipping configuration (secondary block size < 64)..."
            continue
        fi

        if [[ $((s_col * s_row)) -gt 512 ]]; then
            echo "Skipping configuration (secondary block size > 128)..."
            continue
        fi

        for s_cap in "${CAPACITIES[@]}"; do
            # Loop using the indices of the SEEDS array (0 to 4)
            for i in "${!SEEDS[@]}"; do
                seed="${SEEDS[$i]}"
                fold=$i  # The fold is now directly linked to the seed index (0, 1, 2, 3, 4)

                echo "------------------------------------------------"
                echo "Running: cols=$s_col, rows=$s_row, cap=$s_cap, seed=$seed, fold=$fold"
                echo "------------------------------------------------"

                uv run python src/train.py \
                    epochs=100 \
                    dataset.name=cifar10 \
                    seed=$seed \
                    fold=$fold \
                    wandb.enabled=true \
                    "wandb.tags=[\"r:${s_row} c:${s_col} cap:${s_cap}\"]" \
                    wandb.project=momos2d-hierarchical2 \
                    wandb.name="momos_hier_r${s_row}_c${s_col}_cap${s_cap}" \
                    "metrics=[sparsity,l2,gzip,bz2,lzma,bdm]" \
                    quantization=hierarchical_momos2d \
                    quantization.switch_fraction=$SWITCH \
                    quantization.primary.rows=1 \
                    quantization.primary.cols=2 \
                    quantization.primary.capacity=0.01 \
                    quantization.secondary.rows=$s_row \
                    quantization.secondary.cols=$s_col \
                    quantization.secondary.capacity=$s_cap
            done
        done
    done
done
