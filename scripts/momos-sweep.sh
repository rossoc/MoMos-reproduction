echo "========================================="
echo " MoMos Sweep: s ∈ {2,4,8,16,32,64,128}"
echo "              capacity ∈ {0.01,0.05,0.1,0.2,0.3}"
echo "              35 total runs"
echo "========================================="

for S in 2 4 8 16 32 64 128; do
  for CAP in 0.01 0.05 0.1 0.2 0.3; do
    uv run python src/train.py \
      epochs=200 accelerator=cuda dataset.name=cifar10 \
      wandb.enabled=true \
      metrics=[sparsity,l2,gzip,bz2,lzma,bdm] \
      quantization.enabled=true quantization.method=momos \
      quantization.s=$S quantization.capacity=$CAP \
      quantization.force_zero=true quantization.q=32
  done
done

echo ""
echo "========================================="
echo " QAT Sweep: q ∈ {4,8,16}"
echo "              3 total runs"
echo "========================================="


for Q in 4 8 16; do
  uv run python src/train.py \
    epochs=200 accelerator=cuda dataset.name=cifar10 \
    wandb.enabled=true \
    metrics=[sparsity,l2,gzip,bz2,lzma,bdm] \
    quantization.enabled=true quantization.method=qat \
    quantization.q=$Q quantization.exclude_layers=[]
done

echo ""
echo "All sweeps completed."
