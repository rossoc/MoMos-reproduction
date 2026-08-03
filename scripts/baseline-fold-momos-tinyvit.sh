#!/bin/bash
set -e

SEEDS=(42 777 50 78 291)

for i in "${!SEEDS[@]}"; do
    seed="${SEEDS[$i]}"
    fold=$i

    echo "------------------------------------------------"
    echo "Running MoMos Baseline: seed=$seed, fold=$fold"
    echo "------------------------------------------------"

    uv run python src/train.py \
        epochs=200 \
        dataset.name=cifar10 \
        prefix=tinyVit_cifar10 \
        model=tinyvit \
        seed=$seed \
        fold=$fold \
        wandb.enabled=true \
        wandb.project=fold-momos-tinyvit \
        wandb.name="momos-baseline" \
        "metrics=[sparsity,l2,gzip,bz2,lzma,bdm,qbdm]" \
        quantization=momos2d \
        quantization.rows=1 \
        quantization.cols=2 \
        quantization.capacity=0.001

    echo "------------------------------------------------"
    echo "Running Baseline: seed=$seed, fold=$fold"
    echo "------------------------------------------------"
done

for i in "${!SEEDS[@]}"; do
    seed="${SEEDS[$i]}"
    fold=$i

    uv run python src/train.py \
        epochs=200 \
        dataset.name=cifar10 \
        prefix=tinyVit_cifar10 \
        model=tinyvit \
        seed=$seed \
        fold=$fold \
        wandb.enabled=true \
        wandb.project=fold-momos-tinyvit \
        wandb.name="baseline" \
        "metrics=[sparsity,l2,gzip,bz2,lzma,bdm,qbdm]" \
        quantization=none
done
