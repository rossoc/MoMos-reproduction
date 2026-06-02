"""Optuna hyperparameter search for the LitMamba hypernetwork.

Runs an Optuna study that samples Mamba architecture and optimizer
hyperparameters, calls ``train_hypernet.run_training`` for each trial, and
records results in a SQLite-backed study so trials are resumable and
parallelizable.
"""

import copy
import os

import hydra
import optuna
import torch
from omegaconf import DictConfig, OmegaConf
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from train_hypernet import run_training


def _suggest_hparams(trial: optuna.Trial, search_space: DictConfig) -> dict:
    """Sample one hyperparameter configuration from the search-space spec."""
    lr_spec = search_space.learning_rate
    wd_spec = search_space.weight_decay
    nhl_spec = search_space.num_hidden_layers

    return {
        "learning_rate": trial.suggest_float(
            "learning_rate", float(lr_spec.low), float(lr_spec.high), log=bool(lr_spec.log)
        ),
        "weight_decay": trial.suggest_float(
            "weight_decay", float(wd_spec.low), float(wd_spec.high), log=bool(wd_spec.log)
        ),
        "hidden_size": trial.suggest_categorical(
            "hidden_size", list(search_space.hidden_size)
        ),
        "num_heads": trial.suggest_categorical(
            "num_heads", list(search_space.num_heads)
        ),
        "num_hidden_layers": trial.suggest_int(
            "num_hidden_layers", int(nhl_spec.low), int(nhl_spec.high)
        ),
        "state_size": trial.suggest_categorical(
            "state_size", list(search_space.state_size)
        ),
        "n_groups": trial.suggest_categorical(
            "n_groups", list(search_space.n_groups)
        ),
    }


def _apply_overrides(cfg: DictConfig, hparams: dict, trial_epochs: int) -> DictConfig:
    """Return a deep-copied cfg with sampled hyperparameters applied."""
    trial_cfg = copy.deepcopy(cfg)
    trial_cfg.epochs = int(trial_epochs)

    trial_cfg.model.learning_rate = hparams["learning_rate"]
    trial_cfg.model.weight_decay = hparams["weight_decay"]
    trial_cfg.model.hidden_size = hparams["hidden_size"]
    trial_cfg.model.num_heads = hparams["num_heads"]
    trial_cfg.model.num_hidden_layers = hparams["num_hidden_layers"]
    trial_cfg.model.state_size = hparams["state_size"]
    trial_cfg.model.n_groups = hparams["n_groups"]
    return trial_cfg


def _objective_factory(cfg: DictConfig):
    tune_cfg = cfg.tune
    search_space = tune_cfg.search_space
    trial_epochs = int(tune_cfg.trial_epochs)

    def objective(trial: optuna.Trial) -> float:
        hparams = _suggest_hparams(trial, search_space)

        expand = int(cfg.model.get("expand", 2))
        if (hparams["hidden_size"] * expand) % hparams["num_heads"] != 0:
            raise optuna.TrialPruned()
        head_dim = (hparams["hidden_size"] * expand) // hparams["num_heads"]
        if head_dim < 8 or head_dim % 8 != 0:
            raise optuna.TrialPruned()
        if hparams["hidden_size"] % hparams["n_groups"] != 0:
            raise optuna.TrialPruned()

        trial_cfg = _apply_overrides(cfg, hparams, trial_epochs)

        if not tune_cfg.get("wandb_per_trial", False):
            trial_cfg.wandb.enabled = False

        trial_cfg.prefix = f"{cfg.prefix}/trial-{trial.number}"

        try:
            return run_training(trial_cfg, optuna_trial=trial)
        except optuna.TrialPruned:
            raise
        except (RuntimeError, ValueError) as exc:
            msg = str(exc).lower()
            is_oom = "out of memory" in msg or ("cuda" in msg and "memory" in msg)
            is_stride = (
                "multiples of 8" in msg
                or "causal_conv1d" in msg
                or "strides" in msg and "channel last" in msg
            )
            if is_oom or is_stride:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise optuna.TrialPruned() from exc
            raise

    return objective


@hydra.main(config_path="configs", config_name="tune_hypernet", version_base="1.3")
def main(cfg: DictConfig):
    tune_cfg = cfg.tune
    storage_dir = str(tune_cfg.storage_dir)
    os.makedirs(storage_dir, exist_ok=True)

    study_name = str(tune_cfg.study_name)
    storage_url = f"sqlite:///{os.path.join(storage_dir, study_name + '.db')}"

    sampler = TPESampler(seed=int(tune_cfg.sampler.seed))
    pruner = MedianPruner(
        n_startup_trials=int(tune_cfg.pruner.n_startup_trials),
        n_warmup_steps=int(tune_cfg.pruner.n_warmup_steps),
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction=str(tune_cfg.direction),
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    objective = _objective_factory(cfg)
    study.optimize(objective, n_trials=int(tune_cfg.n_trials), gc_after_trial=True)

    print("Best trial:")
    print(f"  number : {study.best_trial.number}")
    print(f"  value  : {study.best_trial.value}")
    print(f"  params : {study.best_trial.params}")

    csv_path = os.path.join(storage_dir, f"{study_name}.csv")
    study.trials_dataframe().to_csv(csv_path, index=False)
    print(f"Trial history written to {csv_path}")

    best_yaml = OmegaConf.create({"best_params": study.best_trial.params,
                                  "best_value": study.best_trial.value})
    with open(os.path.join(storage_dir, f"{study_name}.best.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(best_yaml))


if __name__ == "__main__":
    main()
