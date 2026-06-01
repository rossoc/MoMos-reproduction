#!/bin/bash
set -e

# Optuna sweep for the LitMamba hypernetwork. Resumable: rerun the same
# command and trials are appended to the same SQLite-backed study.
#
# Required override: checkpoint_path (the quantized MLP to predict masks for).

CHECKPOINT_PATH="${CHECKPOINT_PATH:-???}"
N_TRIALS="${N_TRIALS:-50}"
TRIAL_EPOCHS="${TRIAL_EPOCHS:-15}"
STUDY_NAME="${STUDY_NAME:-hypernet_tpe_v1}"
DATASET="${DATASET:-cifar10}"

uv run python src/tune_hypernet.py \
    checkpoint_path="${CHECKPOINT_PATH}" \
    validation_dataset.name="${DATASET}" \
    tune.study_name="${STUDY_NAME}" \
    tune.n_trials="${N_TRIALS}" \
    tune.trial_epochs="${TRIAL_EPOCHS}"
