"""
Symbolic regression search for model-y2opcdot (rows=2, cols=1, high capacity).

Downloads all 20 checkpoints (v0–v19) from WandB, fits the 3-param baseline
G(i) = A·(exp(B·i^C)−1) on every version, and runs PySR on every version to
find closed-form codebook formulas. Outputs a multi-page PDF showing per-version
Pareto fronts, fits vs. empirical quantiles, residuals, and a final summary page
comparing R² evolution across training.
"""
import os
import sys
import warnings
import numpy as np
import wandb
from scipy.optimize import curve_fit, OptimizeWarning
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.view.weight_distribution import load_model
from src.utils.metrics import flatten_weights
from src.view.figure import Figure

WANDB_PROJECT = "danesinoo-university-of-copenhagen/momos-collapse"
MODEL_NAME = "model-y2opcdot"
VERSIONS = range(0, 20)
OUTPUT_PDF = "symbolic_regression_y2opcdot.pdf"
N_ITERATIONS = 60
MAX_SIZE = 25


# ── helpers ──────────────────────────────────────────────────────────────────

def _quantile_targets(weights):
    magnitudes = np.sort(np.abs(weights))
    idx = np.arange(256, dtype=float)
    targets = np.percentile(magnitudes, np.linspace(0, 100, 256))
    return idx, targets


def _baseline(idx, targets):
    """3-parameter exponential baseline: A·(exp(B·i^C)−1)."""
    def G(i, A, B, C):
        return A * (np.exp(B * np.power(i, C)) - 1)

    p0 = [max(targets[-1], 1e-9) / (np.e - 1), 1.0 / 255, 1.0]
    bounds = ([0, 0, 0.1], [np.inf, np.inf, 10.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        (A, B, C), _ = curve_fit(G, idx, targets, p0=p0, bounds=bounds, maxfev=20000)
    predicted = G(idx, A, B, C)
    r2 = _r2(targets, predicted)
    return predicted, r2, f"A·(exp(B·i^C)−1)  [A={A:.3e} B={B:.3e} C={C:.3f}]"


def _r2(y, y_hat):
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _reconstruction_mse(weights, codebook):
    signs = np.sign(weights)
    mags = np.abs(weights)
    enc = np.argmin(np.abs(mags[:, None] - codebook[None, :]), axis=1)
    return float(np.mean((weights - signs * codebook[enc]) ** 2))


def _run_pysr(idx, targets):
    from pysr import PySRRegressor
    sr = PySRRegressor(
        niterations=N_ITERATIONS,
        maxsize=MAX_SIZE,
        binary_operators=["+", "*", "/", "-"],
        unary_operators=[
            "exp",
            "log",
            "sqrt",
            "inv(x) = 1/x",
        ],
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        populations=20,
        population_size=50,
        parsimony=1e-4,
        warm_start=False,
        verbosity=1,
        random_state=42,
    )
    X = idx.reshape(-1, 1)
    sr.fit(X, targets)
    return sr


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    api = wandb.Api()

    # Download all checkpoints
    print(f"Downloading {MODEL_NAME} v{min(VERSIONS)}–v{max(VERSIONS)} from WandB ...")
    checkpoints = {}
    rows = cols = cap = None
    _config_cache = {}

    for i in VERSIONS:
        artifact = api.artifact(f"{WANDB_PROJECT}/{MODEL_NAME}:v{i}")
        if MODEL_NAME not in _config_cache:
            q_cfg = artifact.logged_by().config.get("quantization", {})
            _config_cache[MODEL_NAME] = q_cfg
        q_cfg = _config_cache[MODEL_NAME]
        if rows is None:
            rows = q_cfg.get("rows")
            cols = q_cfg.get("cols")
            cap = q_cfg.get("capacity")
        d = artifact.download()
        ckpt_path = os.path.join(d, "model.ckpt")
        checkpoints[i] = ckpt_path
        print(f"  v{i}: {ckpt_path}")

    print(f"\nModel config: rows={rows} cols={cols} capacity={cap}")

    # Per-version results
    results = {}  # version -> dict with keys: idx, targets, base_pred, base_r2, base_label,
                  #                              sr_pred, sr_r2, sr_eq, sr_equations, weights

    for i in VERSIONS:
        print(f"\n── Version v{i} ──────────────────────────────────────────────")
        model = load_model(checkpoints[i])
        weights = flatten_weights(model)
        idx, targets = _quantile_targets(weights)

        # Baseline
        base_pred, base_r2, base_label = _baseline(idx, targets)
        base_mse = _reconstruction_mse(weights, base_pred.clip(min=0))
        print(f"Baseline {base_label}")
        print(f"  R²={base_r2:.4f}  reconstruction MSE={base_mse:.4e}")

        # PySR
        print(f"Running PySR (niterations={N_ITERATIONS}, maxsize={MAX_SIZE}) ...")
        sr = _run_pysr(idx, targets)
        print(sr)

        X = idx.reshape(-1, 1)
        best_eq = sr.sympy()
        best_pred = sr.predict(X)
        best_r2 = _r2(targets, best_pred)
        best_mse = _reconstruction_mse(weights, best_pred.clip(min=0))
        print(f"Best expression: {best_eq}")
        print(f"  R²={best_r2:.4f}  reconstruction MSE={best_mse:.4e}")

        results[i] = {
            "idx": idx,
            "targets": targets,
            "weights": weights,
            "base_pred": base_pred,
            "base_r2": base_r2,
            "base_label": base_label,
            "base_mse": base_mse,
            "sr_pred": best_pred,
            "sr_r2": best_r2,
            "sr_eq": best_eq,
            "sr_mse": best_mse,
            "sr_equations": sr.equations_,
        }

    # ── PDF output ────────────────────────────────────────────────────────────
    with PdfPages(OUTPUT_PDF) as pdf:
        for i in VERSIONS:
            r = results[i]
            idx = r["idx"]
            targets = r["targets"]

            # Page A: SR expression vs baseline vs empirical quantiles
            fig = Figure(
                title=f"v{i}: Best symbolic expression vs baseline",
                ncols=1,
                nrows=1,
                figSize=(12, 6),
            )
            fig.plot(
                {
                    "empirical quantiles": (idx, targets),
                    f"baseline  R²={r['base_r2']:.4f}": (idx, r["base_pred"]),
                    f"SR best  R²={r['sr_r2']:.4f}": (idx, r["sr_pred"]),
                },
                exp_name=f"SR: {r['sr_eq']}",
                x_label="index i",
                y_label="|w|",
            )
            fig.save(pdf=pdf)

            # Page B: Residuals comparison
            fig2 = Figure(
                title=f"v{i}: Residuals — baseline vs SR",
                ncols=2,
                nrows=1,
                figSize=(14, 5),
            )
            fig2.plot(
                {"baseline residuals": (idx, targets - r["base_pred"])},
                exp_name=f"Baseline residuals  (R²={r['base_r2']:.4f})",
                x_label="index i",
                y_label="residual",
                legend=False,
            )
            fig2.plot(
                {"SR residuals": (idx, targets - r["sr_pred"])},
                exp_name=f"SR residuals  (R²={r['sr_r2']:.4f})",
                x_label="index i",
                y_label="residual",
                legend=False,
            )
            fig2.save(pdf=pdf)

            # Page C: Pareto front
            equations = r["sr_equations"]
            complexities = equations["complexity"].tolist()
            losses = equations["loss"].tolist()
            fig3 = Figure(
                title=f"v{i}: Symbolic regression Pareto front",
                ncols=1,
                nrows=1,
                figSize=(10, 6),
            )
            fig3.plot(
                {"complexity vs loss": (complexities, losses)},
                exp_name="Pareto front  (lower-left = better)",
                x_label="expression complexity",
                y_label="MSE loss",
                symbol="o-",
                legend=False,
            )
            fig3.save(pdf=pdf)

        # Summary page: R² evolution across versions
        versions = list(VERSIONS)
        base_r2s = [results[i]["base_r2"] for i in versions]
        sr_r2s = [results[i]["sr_r2"] for i in versions]

        fig_summary = Figure(
            title=f"R² evolution across training ({MODEL_NAME}  rows={rows} cols={cols} cap={cap})",
            ncols=1,
            nrows=1,
            figSize=(12, 6),
        )
        fig_summary.plot(
            {
                "baseline R²": (versions, base_r2s),
                "SR best R²": (versions, sr_r2s),
            },
            exp_name="R² vs checkpoint version",
            x_label="version",
            y_label="R²",
            symbol="o-",
        )
        fig_summary.save(pdf=pdf)

    print(f"\nSaved to {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
