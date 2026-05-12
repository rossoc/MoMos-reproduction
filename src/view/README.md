# src/view

Visualization and reporting layer for MoMos experiments. Fetches run data from WandB, computes compression metrics, analyzes learned weight distributions, and assembles everything into PDF reports.

## Files

| File | Purpose |
|------|---------|
| `fetch_log.py` | Fetches runs from WandB; cleans and returns training metrics as DataFrames |
| `compression_metrics.py` | Computes RAC (Relative Algorithmic Compression) and BDM complexity from run configs |
| `weight_distribution.py` | Loads model checkpoints, extracts learned motifs/blocks, plots frequencies, norms, and correlations |
| `figure.py` | `Figure` class — thin matplotlib wrapper for consistent publication-quality plots |
| `report.py` | `Report` class — assembles multi-figure PDFs from training curves, 3D contour plots, and weight analysis |
| `__init__.py` | Re-exports the public API |

## Data flow

```
WandB API ──► fetch_log.py ──────────────► report.py ──► PDF
                                               ▲
              compression_metrics.py ──────────┤
              weight_distribution.py ──────────┘

              figure.py  (used by all plotting code)
```

## Key abstractions

**`Figure`** (`figure.py`) wraps a matplotlib figure with named subplots. Supports line plots, variance bands, dual-axis plots, and 3D contour plots. Call `.save()` with a `PdfPages` object to write to a shared PDF.

**`Report`** (`report.py`) holds a list of `Figure` objects and writes them all to a single PDF via `.save(filename)`. `Report.training_overview()` and `Report.metrics_vs_accuracy()` are the main entry points for building standard report pages.

**`fetch_runs`** (`fetch_log.py`) returns a list of run dicts `{name, metrics, config, test_acc}` ready to pass into `Report` methods.

**`compute_compression_metrics`** (`compression_metrics.py`) takes a run dict and returns `{rac, rac_qat, bdm_complexity, bdm_ratio}`.

**`plot_weights` / `plot_weights_2d`** (`weight_distribution.py`) take a checkpoint path and block dimensions and return a list of `Figure` objects showing motif frequency, L2 norms, and per-layer scatter plots.
