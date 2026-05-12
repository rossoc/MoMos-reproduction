#!/bin/bash
COLS=(1 2 4)
ROWS=(1 2 4)
# SEEDS=(42 123 777 482 291)

# Iterate through all combinations
for col in "${COLS[@]}"; do
    for row in "${ROWS[@]}"; do
        
        # Skip the case where it's a 1x1 grid
        if [[ $col -eq 1 && $row -eq 1 ]]; then
            echo "Skipping 1x1 configuration..."
            continue
        fi

        # for seed in "${SEEDS[@]}"; do

            echo "------------------------------------------------"
            echo "Running: cols=$col, rows=$row" #, seed=$seed"
            echo "------------------------------------------------"

            uv run python src/train.py \
                epochs=100 \
                accelerator=cuda \
                dataset.name=cifar10 \
                wandb.enabled=true \
                wandb.project="momos-static" \
                wandb.name="c${col}_r${row}_cap${cap}_s${seed}" \
                "metrics=[sparsity,l2,gzip,bz2,lzma,bdm]" \
                quantization.enabled=true \
                quantization.method=static_momos2d \
                quantization.cols=$col \
                quantization.rows=$row \
                quantization.k=256 \
                quantization.force_zero=true \
                quantization.init_mode=sr_rational \
                quantization.q=32 \
                seed=42 # $seed

        # done
    done
done
