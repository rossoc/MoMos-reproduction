"""Lightning callbacks for quantization-aware training and MoMos projection."""

import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from omegaconf import OmegaConf

from quantizers import quantize_qat, k_from_capacity, quantize
from utils.metrics import compute_metrics


class QuantizationCallback(L.Callback):
    """Callback that handles QAT setup and MoMos projection during training.

    This callback integrates quantization into the Lightning training lifecycle:
    - QAT: Attaches fake-quant parametrizations before training starts
    - MoMos: Applies motif-based projection at the end of each training epoch

    Args:
        quant_cfg: Quantization configuration dictionary. Must contain ``method`` key
            with value ``"qat"`` or ``"momos"``. Other keys depend on the method:
            - QAT: ``q`` (bit-width), ``exclude_layers`` (optional)
            - MoMos: ``s`` (block size), ``k`` (motif count), ``force_zero``, etc.
        metric_names: Optional list of metric names to compute and log after each
            epoch. Uses ``utils.metrics.compute_metrics``. Examples: ``sparsity``,
            ``l2``, ``gzip``, ``bz2``, ``lzma``, ``bdm``.
        compression_binarized: Passed to metric computation for compression payloads.
    """

    def __init__(
        self,
        quant_cfg: dict,
        metric_names: list[str] | None = None,
        compression_binarized: bool = False,
    ):
        super().__init__()
        self.quant_cfg = quant_cfg
        self.metric_names = metric_names or []
        self.compression_binarized = compression_binarized
        self.method = str(quant_cfg.get("method", "")).lower()

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Set up QAT parametrizations before training begins."""
        if self.method != "qat":
            return

        model = pl_module.model
        stats = quantize_qat(model, self.quant_cfg)

        qat_enabled = stats.get("qat_enabled", False)
        attached = stats.get("attached_modules", 0)
        updated = stats.get("updated_modules", 0)

        if qat_enabled:
            print(
                f"QAT enabled: attached={attached}, updated={updated}, "
                f"bitwidth={stats.get('q_bits', '?')}"
            )
        else:
            print(
                f"QAT disabled (bitwidth >= 32): disabled_modules={stats.get('disabled_modules', 0)}"
            )

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Apply MoMos projection at the end of each training epoch."""
        if self.method not in ["momos", "momos2d", "static_momos2d"]:
            return

        model = pl_module.model

        # Resolve k from capacity if needed
        if (
            self.quant_cfg.get("k") is None
            and self.quant_cfg.get("capacity") is not None
        ):
            self.quant_cfg["k"] = k_from_capacity(
                model, self.quant_cfg["s"], self.quant_cfg["capacity"]
            )

        stats = quantize(model, self.quant_cfg)

        report = ""
        for k, v in stats.items():
            if k in ["motif_counts", "method"]:
                continue
            pl_module.log("quant/" + k, v, on_epoch=True, prog_bar=False)
            report += f"{k}={v:.4f}"

        print("MoMos applied:", report)

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Compute and log quantization metrics after validation."""
        if not self.metric_names:
            return

        model = pl_module.model
        try:
            rows = int(self.quant_cfg.get("rows") or 1)
            cols = int(self.quant_cfg.get("cols") or 1)
            metrics = compute_metrics(
                model,
                self.metric_names,
                self.compression_binarized,
                rows=rows,
                cols=cols,
            )
            for name, value in metrics.items():
                if value is not None:
                    pl_module.log(
                        f"metrics/{name}", value, on_epoch=True, prog_bar=False
                    )
        except Exception as e:
            print(f"Warning: Failed to compute metrics: {e}")


def _best_checkpoint(checkpoint_dir: str) -> ModelCheckpoint:
    return ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=False,
    )


def _periodic_checkpoint(checkpoint_dir: str) -> ModelCheckpoint:
    return ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="epoch-{epoch:02d}",
        every_n_epochs=20,
        save_top_k=-1,
    )


def _early_stopping(patience: int | None) -> EarlyStopping | None:
    if patience is None or patience <= 0:
        return None
    return EarlyStopping(
        monitor="val/loss", mode="min", patience=patience, verbose=True
    )


def _build_quantization_callback(cfg, quant_cfg) -> QuantizationCallback:
    """Convert a per-method `quantization` config into a QuantizationCallback.

    The YAML file selected by Hydra's `quantization=<method>` group already
    contains exactly the keys the method consumes, so we forward the dict
    as-is (minus the config-only `enabled` flag).
    """
    full = OmegaConf.to_container(quant_cfg, resolve=True)
    assert isinstance(full, dict)
    full.pop("enabled", None)

    method = full.get("method")
    if method in ("momos2d", "static_momos2d"):
        full["s"] = int(full["rows"]) * int(full["cols"])

    if method in ("momos", "momos2d", "static_momos2d"):
        if full.get("k") is None and full.get("capacity") is None:
            raise ValueError(f"{method} requires either k or capacity in config")

    return QuantizationCallback(
        quant_cfg=full,
        metric_names=cfg.get("metrics", []),
        compression_binarized=cfg.get("all_compression_metrics_binarized", False),
    )


def build_callbacks(
    cfg,
    checkpoint_dir: str,
    unique_run_name: str,
    has_logger: bool = True,
) -> list[L.Callback]:
    """Build the complete list of callbacks for training.

    Includes standard callbacks (checkpointing, early stopping, LR monitoring)
    plus a quantization callback if `cfg.quantization.enabled` is true. The
    per-method quantization schema lives in `src/configs/quantization/`.
    """
    callbacks: list[L.Callback] = []

    if has_logger:
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    callbacks.append(_best_checkpoint(checkpoint_dir))

    if cfg.get("periodic_checkpoint"):
        callbacks.append(_periodic_checkpoint(checkpoint_dir))

    if (es := _early_stopping(cfg.get("patience"))) is not None:
        callbacks.append(es)

    quant_cfg = cfg.get("quantization", {})
    if quant_cfg.get("enabled", False):
        callbacks.append(_build_quantization_callback(cfg, quant_cfg))

    return callbacks
