"""Lightning callbacks for quantization-aware training and MoMos projection."""

import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)

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
        if self.method not in ["momos", "momos2d"]:
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
            pl_module.log("quant/" + k, v, on_epoch=True, prog_bar=False)
            report += f"{k}={v:.4f}"

        print("MoMos applied:", report)

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Compute and log quantization metrics after validation."""
        if not self.metric_names:
            return

        model = pl_module.model
        try:
            metrics = compute_metrics(
                model, self.metric_names, self.compression_binarized
            )
            for name, value in metrics.items():
                if value is not None:
                    pl_module.log(
                        f"metrics/{name}", value, on_epoch=True, prog_bar=False
                    )
        except Exception as e:
            print(f"Warning: Failed to compute metrics: {e}")


def build_callbacks(
    cfg,
    checkpoint_dir: str,
    unique_run_name: str,
    has_logger: bool = True,
) -> list[L.Callback]:
    """Build the complete list of callbacks for training.

    This factory function creates all standard callbacks (checkpointing, early stopping,
    LR monitoring) plus quantization callbacks if enabled in the config.

    Args:
        cfg: Full Hydra configuration dictionary. Expected structure:
            - ``cfg.patience``: Early stopping patience (or None)
            - ``cfg.quantization.enabled``: Whether quantization is active
            - ``cfg.quantization.method``: ``"qat"`` or ``"momos"``
            - ``cfg.quantization.*``: Method-specific parameters
            - ``cfg.metrics``: List of metric names to track
        checkpoint_dir: Directory path for saving model checkpoints.
        unique_run_name: Unique identifier for this run (used in checkpoint naming).
        has_logger: Whether a logger (e.g. W&B) will be used. Controls whether
            ``LearningRateMonitor`` is included.

    Returns:
        List of Lightning callback instances ready for ``Trainer(callbacks=...)``.
    """
    callbacks: list[L.Callback] = []

    # Learning rate monitor (requires a logger)
    if has_logger:
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=False,
    )
    callbacks.append(checkpoint_callback)

    periodic_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="epoch-{epoch:02d}",  # Includes epoch number in filename
        every_n_epochs=20,
        save_top_k=-1,  # Set to -1 to keep all periodic checkpoints
        # Or set to 3 to keep only the last 3 periodic ones
    )
    callbacks.append(periodic_checkpoint)

    # Early stopping (optional)
    patience = cfg.get("patience")
    if patience is not None and patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor="val/loss",
                mode="min",
                patience=patience,
                verbose=True,
            )
        )

    # Quantization callback (if enabled)
    quant_cfg = cfg.get("quantization", {})
    if quant_cfg.get("enabled", False):
        method = quant_cfg.get("method")
        if method is not None and method.lower() in ("qat", "momos", "momos2d"):
            # Build full quantization config dict for the quantizer modules
            full_quant_cfg = {
                "method": method.lower(),
                "q": int(quant_cfg.get("q", 32)),
            }
            if method.lower() == "qat":
                full_quant_cfg["exclude_layers"] = quant_cfg.get("exclude_layers", [])
            elif method.lower() == "momos":
                full_quant_cfg["s"] = int(quant_cfg["s"])
                # Resolve k from direct value or capacity
                if quant_cfg.get("k") is not None:
                    full_quant_cfg["k"] = int(quant_cfg["k"])
                elif quant_cfg.get("capacity") is not None:
                    # Will be resolved by callback using model at epoch end
                    full_quant_cfg["capacity"] = float(quant_cfg["capacity"])
                    full_quant_cfg["k"] = None  # placeholder, resolved in callback
                else:
                    raise ValueError("MoMos requires either k or capacity in config")
                full_quant_cfg["force_zero"] = bool(quant_cfg.get("force_zero", True))
                if "chunk_size" in quant_cfg:
                    full_quant_cfg["chunk_size"] = quant_cfg["chunk_size"]
                if "chunk_progress" in quant_cfg:
                    full_quant_cfg["chunk_progress"] = bool(quant_cfg["chunk_progress"])
                if "chunk_progress_elements" in quant_cfg:
                    full_quant_cfg["chunk_progress_elements"] = quant_cfg[
                        "chunk_progress_elements"
                    ]
                if (
                    quant_cfg.get("from_percentile") is not None
                    and quant_cfg.get("to_percentile") is not None
                    and quant_cfg.get("swapping_probability") is not None
                ):
                    full_quant_cfg |= {
                        "from_percentile": quant_cfg.get("from_percentile"),
                        "to_percentile": quant_cfg.get("to_percentile"),
                        "swapping_probability": quant_cfg.get("swapping_probability"),
                    }
            elif method.lower() == "momos2d":
                full_quant_cfg["cols"] = int(quant_cfg["rows"])
                # Resolve k from direct value or capacity
                if quant_cfg.get("k") is not None:
                    full_quant_cfg["k"] = int(quant_cfg["k"])
                elif quant_cfg.get("capacity") is not None:
                    # Will be resolved by callback using model at epoch end
                    full_quant_cfg["capacity"] = float(quant_cfg["capacity"])
                    full_quant_cfg["k"] = None  # placeholder, resolved in callback
                else:
                    raise ValueError("MoMos requires either k or capacity in config")
                full_quant_cfg["force_zero"] = bool(quant_cfg.get("force_zero", True))
                if "chunk_size" in quant_cfg:
                    full_quant_cfg["chunk_size"] = quant_cfg["chunk_size"]
                if "chunk_progress" in quant_cfg:
                    full_quant_cfg["chunk_progress"] = bool(quant_cfg["chunk_progress"])
                if "chunk_progress_elements" in quant_cfg:
                    full_quant_cfg["chunk_progress_elements"] = quant_cfg[
                        "chunk_progress_elements"
                    ]
                if (
                    quant_cfg.get("from_percentile") is not None
                    and quant_cfg.get("to_percentile") is not None
                    and quant_cfg.get("swapping_probability") is not None
                ):
                    full_quant_cfg |= {
                        "from_percentile": quant_cfg.get("from_percentile"),
                        "to_percentile": quant_cfg.get("to_percentile"),
                        "swapping_probability": quant_cfg.get("swapping_probability"),
                    }

            callbacks.append(
                QuantizationCallback(
                    quant_cfg=full_quant_cfg,
                    metric_names=cfg.get("metrics", []),
                    compression_binarized=cfg.get(
                        "all_compression_metrics_binarized", False
                    ),
                )
            )

    return callbacks
