"""Force a `method: none` (dense/no-quantization) baseline trial into an
existing Optuna quantization study whose `optuna.methods` never included
"none" - without touching that parameter's already-locked choice set.

Optuna locks a categorical parameter's choice set to whatever the first
trial that used that parameter name recorded, for the life of the study.
Widening `optuna.methods` to include "none" and resuming - even by forcing
the value via `study.enqueue_trial({"method": "none"})` - raises:

    ValueError: CategoricalDistribution does not support dynamic value space.

This script sidesteps that by suggesting the forced trial's method under a
fresh parameter name (`forced_baseline_method`, never reused by any other
trial) instead of `method`. Every downstream reader
(`study.best_trials`, `study.trials_dataframe()`,
`notebook/pareto_front_analysis.py`) only looks at
`trial.user_attrs["method"]`, which this still sets to "none" - so the
baseline merges into the existing study/db seamlessly, and every other
trial's `method` distribution is left completely untouched.

The trial is produced by the exact same pipeline as every other trial in
the study (`build_backbone`, `compute_quantization_bits`,
`_build_datamodule`, `run_training`) so it's a genuine, comparable point on
the Pareto front - not an estimate.

Usage: reuse whatever Hydra overrides produced the target study (same
config group as tune_quantization.py), just swap the entry point:

    python src/inject_baseline_trial.py model=resnet20 \
        optuna.study_name=resnet_paretofront4 optuna.storage_dir=./outputs
"""

import copy
import gc
import os
import statistics

import hydra
import optuna
from omegaconf import DictConfig, OmegaConf, open_dict

from model import build_backbone
from train import run_training
from tune_quantization import _build_datamodule, _bump_fd_limit
from utils.quant_bits import compute_quantization_bits

# Never reused for any other parameter/trial in the study - this name is the
# whole trick: it can't collide with the locked `method` CategoricalDistribution.
_BASELINE_PARAM_NAME = "forced_baseline_method"


def _already_injected(study: optuna.Study) -> bool:
    """True if a COMPLETE method='none' trial already exists in `study`."""
    return any(
        t.user_attrs.get("method") == "none"
        for t in study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
    )


def _baseline_objective_factory(cfg: DictConfig, backbone_sample):
    """Mirrors the method == "none" branch of tune_quantization.py's
    `_objective_factory.objective`, minus the exact-reuse cache (there is
    nothing to reuse - this trial is one-of-a-kind by construction)."""

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        # Fresh param name -> never conflicts with the study's locked `method`
        # distribution, unlike `trial.suggest_categorical("method", methods)`.
        trial.suggest_categorical(_BASELINE_PARAM_NAME, ["none"])

        quant_cfg = {"enabled": False, "method": "none"}
        bit_stats = compute_quantization_bits(backbone_sample, quant_cfg)
        compression_rate = float(bit_stats["compression_rate"])

        trial.set_user_attr("compression_rate", compression_rate)
        trial.set_user_attr("compressed_bits", float(bit_stats["compressed_bits"]))
        trial.set_user_attr("bpp", float(bit_stats["bpp"]))
        trial.set_user_attr("method", "none")

        trial_cfg = copy.deepcopy(cfg)
        with open_dict(trial_cfg):
            trial_cfg.quantization = OmegaConf.create(quant_cfg)
            trial_cfg.prefix = f"{cfg.prefix}/baseline-trial-{trial.number}"

        opt_cfg = cfg.optuna
        search_folds = list(opt_cfg.get("eval_folds_search", [0]))
        val_accs = []
        for fold in search_folds:
            fold_cfg = copy.deepcopy(trial_cfg)
            with open_dict(fold_cfg):
                fold_cfg.fold = int(fold)
                if not cfg.get("wandb", {}).get("enabled", False):
                    fold_cfg.wandb.enabled = False
                else:
                    fold_cfg.wandb.name = f"trial-{trial.number}_none_f{fold}"

            dm = None
            res = None
            try:
                dm = _build_datamodule(cfg, fold)
                res = run_training(fold_cfg, optuna_trial=trial, datamodule=dm)
                val_accs.append(res["val_acc"])
            except optuna.TrialPruned:
                raise
            except (RuntimeError, ValueError) as exc:
                msg = str(exc).lower()
                if (
                    "out of memory" in msg
                    or "cuda" in msg
                    or "too many open files" in msg
                    or "errno 24" in msg
                ):
                    raise optuna.TrialPruned() from exc
                raise
            finally:
                del dm, res
                gc.collect()

        mean_val_acc = statistics.mean(val_accs) if val_accs else 0.0
        trial.set_user_attr("val_acc", mean_val_acc)
        return mean_val_acc, compression_rate

    return objective


@hydra.main(config_path="configs", config_name="tune_quantization", version_base="1.3")
def main(cfg: DictConfig):
    opt_cfg = cfg.optuna
    storage_dir = str(opt_cfg.storage_dir)
    study_name = str(opt_cfg.study_name)
    storage_url = f"sqlite:///{os.path.join(storage_dir, study_name + '.db')}"

    print(f"\n[inject_baseline_trial] loading study '{study_name}' from {storage_url}")
    study = optuna.load_study(study_name=study_name, storage=storage_url)

    if _already_injected(study):
        print(
            f"[inject_baseline_trial] study '{study_name}' already has a COMPLETE "
            "method='none' trial - nothing to do."
        )
        return

    img_size = cfg.model.get("img_size", None) or cfg.dataset.img_size
    backbone_sample = build_backbone(
        cfg.model,
        in_channels=cfg.dataset.in_channels,
        img_size=img_size,
        num_classes=cfg.dataset.num_classes,
    )

    _bump_fd_limit()

    print(f"[inject_baseline_trial] forcing one method='none' baseline trial into '{study_name}'\n")
    study.enqueue_trial({_BASELINE_PARAM_NAME: "none"})
    objective = _baseline_objective_factory(cfg, backbone_sample)
    study.optimize(objective, n_trials=1)

    forced = study.trials[-1]
    print(
        f"\n[inject_baseline_trial] trial #{forced.number:03d} | method: none | "
        f"Val Acc: {forced.values[0]:.4f} | Comp Rate: {forced.values[1]:.2f}x"
    )


if __name__ == "__main__":
    main()
