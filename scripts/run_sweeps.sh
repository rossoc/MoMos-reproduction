#!/usr/bin/env bash
set -euo pipefail

# MoMos + QAT sweep script
# Runs Hydra --multirun sweeps over MoMos (s × capacity) and QAT (q) configs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================="
echo " MoMos Sweep: s ∈ {2,4,8,16,32,64,128}"
echo "              capacity ∈ {0.01,0.05,0.1,0.2,0.3}"
echo "              35 total runs"
echo "========================================="

uv run python src/train.py \
  --multirun \
  epochs=200 \
  accelerator=cuda \
  dataset.name=cifar10 \
  wandb.enabled=true \
  metrics=[sparsity,l2,gzip,bz2,lzma,bdm] \
  quantization.enabled=true \
  quantization.method=momos \
  quantization.s=2,4,8,16,32,64,128 \
  quantization.capacity=0.01,0.05,0.1,0.2,0.3 \
  quantization.force_zero=true \
  quantization.q=32

echo ""
echo "========================================="
echo " QAT Sweep: q ∈ {4,8,16}"
echo "              3 total runs"
echo "========================================="

uv run python src/train.py \
  --multirun \
  epochs=200 \
  accelerator=cuda \
  dataset.name=cifar10 \
  wandb.enabled=true \
  metrics=[sparsity,l2,gzip,bz2,lzma,bdm] \
  quantization.enabled=true \
  quantization.method=qat \
  quantization.q=4,8,16 \
  quantization.exclude_layers=[]

echo ""
echo "All sweeps completed."
