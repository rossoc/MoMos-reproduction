"""Lightning callbacks for quantization-aware training and MoMos projection."""

import math

import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from omegaconf import OmegaConf

from quantizers import quantize_qat, k_from_capacity, quantize
from quantizers.block_utils import iter_trainable_params
from utils.metrics import compute_metrics


def _hierarchical_bit_estimate(n, k, s, s_prime, k_per_bucket, q):
    """Bit cost of the {Z, iota, M'} description (rounded up per integer index).

    motifs: k * s * q
    iota:   s' * k_per_bucket * ceil(log2 k)
    mosaic: ceil(n / s) ... wait — n/(s*s') big-block indices, each ceil(log2 k')
            with k' = k_per_bucket ** s', so ceil(s' * log2 k_per_bucket) bits.
    """
    motif_bits = k * s * q
    idx_bits = math.ceil(math.log2(max(2, k)))
    iota_bits = s_prime * k_per_bucket * idx_bits
    n_big = math.ceil(n / (s * s_prime))
    big_idx_bits = math.ceil(s_prime * math.log2(max(2, k_per_bucket)))
    mosaic_bits = n_big * big_idx_bits
    total = motif_bits + iota_bits + mosaic_bits
    return {
        "motif_bits": motif_bits,
        "iota_bits": iota_bits,
        "mosaic_bits": mosaic_bits,
        "total_bits": total,
    }


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
        if self.method == "hierarchical_momos2d":
            self._log_hierarchical_bit_estimate(pl_module)
            return

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

    def _log_hierarchical_bit_estimate(self, pl_module: L.LightningModule):
        model = pl_module.model
        primary = self.quant_cfg["primary"]
        secondary = self.quant_cfg["secondary"]

        s = int(primary["s"])
        s_prime = int(secondary["s"])
        q = int(self.quant_cfg.get("q_bits", 32))
        c = float(secondary["capacity"])

        if primary.get("k") is None and primary.get("capacity") is not None:
            primary["k"] = k_from_capacity(model, s, primary["capacity"])
        k = int(primary["k"])

        n = sum(p.numel() for p in iter_trainable_params(model))
        n_big = math.ceil(n / (s * s_prime))
        # Bucket size is relative to K_primary (capacity * K), matching the
        # quantizer in momos_hierarchy.py; capped at the big-block count.
        k_per_bucket = max(1, min(int(round(k * c)), n_big))

        est = _hierarchical_bit_estimate(n, k, s, s_prime, k_per_bucket, q)
        baseline = n * q
        print(
            f"Hierarchical MoMos2D bit estimate (q={q}, n={n}, k={k}, s={s}, "
            f"s'={s_prime}, k_per_bucket={k_per_bucket}): "
            f"motif={est['motif_bits']}, iota={est['iota_bits']}, "
            f"mosaic={est['mosaic_bits']}, total={est['total_bits']} "
            f"({est['total_bits'] / baseline:.4%} of dense {baseline} bits)"
        )

        logger = getattr(pl_module, "logger", None)
        if logger is not None and hasattr(logger, "log_hyperparams"):
            try:
                logger.log_hyperparams(
                    {
                        "bits/motif": est["motif_bits"],
                        "bits/iota": est["iota_bits"],
                        "bits/mosaic": est["mosaic_bits"],
                        "bits/total": est["total_bits"],
                        "bits/dense": baseline,
                        "bits/ratio": est["total_bits"] / baseline,
                    }
                )
            except Exception:
                pass

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Apply MoMos projection at the end of each training epoch."""
        if self.method not in [
            "momos",
            "momos2d",
            "static_momos2d",
            "hierarchical_momos2d",
        ]:
            return

        model = pl_module.model

        if self.method == "hierarchical_momos2d":
            switch_fraction = float(self.quant_cfg.get("switch_fraction", 0.5))
            max_epochs = max(1, int(trainer.max_epochs or 1))
            progress = trainer.current_epoch / max_epochs
            self.quant_cfg["apply_secondary"] = progress >= switch_fraction

            # Only primary uses k_from_capacity: secondary `capacity` is interpreted
            # relative to K_primary inside the hierarchy module.
            primary = self.quant_cfg["primary"]
            if primary.get("k") is None and primary.get("capacity") is not None:
                primary["k"] = k_from_capacity(
                    model, primary["s"], primary["capacity"]
                )
        else:
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
        skip = {
            "motif_counts",
            "method",
            "iota",
            "bucket_sizes",
            "K_primary",
            "s_prime",
            "k_per_bucket",
            "_motifs",
            "_nearest",
            "_layer_specs",
        }
        for k, v in stats.items():
            if k in skip:
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
            if self.method == "hierarchical_momos2d":
                active = (
                    self.quant_cfg["secondary"]
                    if self.quant_cfg.get("apply_secondary", False)
                    else self.quant_cfg["primary"]
                )
                rows = int(active.get("rows") or 1)
                cols = int(active.get("cols") or 1)
            else:
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

    if method == "hierarchical_momos2d":
        for sub_key in ("primary", "secondary"):
            sub = full.get(sub_key)
            if not isinstance(sub, dict):
                raise ValueError(
                    f"{method} requires a `{sub_key}` block in the quantization config"
                )
            sub["s"] = int(sub["rows"]) * int(sub["cols"])
            if sub.get("k") is None and sub.get("capacity") is None:
                raise ValueError(
                    f"{method}.{sub_key} requires either k or capacity in config"
                )

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
