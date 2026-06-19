#!/bin/bash
set -e

SEEDS=(42 777 50 78 291)

for i in "${!SEEDS[@]}"; do
    seed="${SEEDS[$i]}"
    fold=$i  # The fold is now directly linked to the seed index (0, 1, 2, 3, 4)

    uv run python src/train.py \
      epochs=200 \
      accelerator=cuda \
      dataset.name=cifar10 \
      wandb.project=momos2d-hierarchical3 \
      wandb.enabled=true \
      wandb.name="baseline-momos" \
      metrics=[sparsity,l2,gzip,bz2,lzma,bdm] \
      quantization=momos \
      quantization.s=2 quantization.capacity=0.001 \
      quantization.force_zero=true
  done

echo "All sweeps completed."
