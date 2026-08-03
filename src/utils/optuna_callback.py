"""Optuna pruning callback that reports after each training epoch.

The default ``PyTorchLightningPruningCallback`` from ``optuna-integration``
hooks into ``on_validation_end``. ``LitMamba`` has no ``validation_step`` —
its ``val/acc`` is logged from inside ``on_train_epoch_end`` — so this
custom callback reports metrics on the same hook the model uses.
"""

from __future__ import annotations

import lightning as L
import optuna


class OptunaTrainEpochPruning(L.Callback):
    """Report ``monitor`` to an Optuna trial after every training epoch.

    Raises ``optuna.TrialPruned`` if the trial should be pruned.
    """

    def __init__(self, trial: optuna.Trial, monitor: str = "val/acc"):
        super().__init__()
        self.trial = trial
        self.monitor = monitor

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        # Optuna trial reporting and pruning are only supported for single-objective studies
        study = self.trial.study
        if len(study.directions) > 1 or getattr(study, "_is_multi_objective", lambda: False)():
            return
        metric = trainer.callback_metrics.get(self.monitor)
        if metric is None:
            return
        value = float(metric.item() if hasattr(metric, "item") else metric)
        epoch = int(trainer.current_epoch)
        self.trial.report(value, step=epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned(
                f"Trial pruned at epoch {epoch} ({self.monitor}={value:.4f})"
            )
