#!/bin/bash

# Removed commas: Bash arrays use spaces as separators
PERCS=("[0.1]" "[0.5]" "[0.95]" "[0.99]")
PROBS=(0.8 1)
SEEDS=(42 50 78 2 23)

# SET THIS TO THE EXPERIMENT NUMBER THAT CRASHED
RESUME_FROM=0
experiment_count=1

for f_perc in "${PERCS[@]}"; do
    for t_perc in "${PERCS[@]}"; do
        
        # Skip if from and to percentiles are the same
        if [[ "$f_perc" == "$t_perc" ]]; then
            continue
        fi

        for prob in "${PROBS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                
                # --- RESUME LOGIC ---
                if [ $experiment_count -lt $RESUME_FROM ]; then
                    ((experiment_count++))
                    continue
                fi
                # --------------------

                # Clean up names for WandB (removes brackets and quotes)
                clean_f=$(echo "$f_perc" | tr -d '[]"')
                clean_t=$(echo "$t_perc" | tr -d '[]"')

                echo "-------------------------------------------------------"
                echo "Running Experiment #$experiment_count"
                echo "Mapping: $f_perc -> $t_perc | Prob: $prob | Seed: $seed"
                echo "-------------------------------------------------------"

                # Added quotes around metrics to prevent shell expansion issues
                uv run python src/train.py \
                    epochs=500 \
                    accelerator=cuda \
                    dataset.name=cifar10 \
                    wandb.enabled=true \
                    wandb.name="momos_swapping_${clean_f}_t${clean_t}_p${prob}_s${seed}" \
                    "metrics=[sparsity,l2,gzip,bz2,lzma,bdm]" \
                    quantization.enabled=true \
                    quantization.method=momos \
                    quantization.s=2 \
                    quantization.capacity=0.01 \
                    quantization.force_zero=true \
                    quantization.q=32 \
                    quantization.swapping_probability="$prob" \
                    quantization.from_percentile="$f_perc" \
                    quantization.to_percentile="$t_perc" \
                    seed="$seed"

                echo "Experiment #$experiment_count complete."
                ((experiment_count++))
                echo ""
            done
        done
    done
done
