"""
Symbolic regression search for a closed-form codebook function G(i).

Uses PySR to search over combinations of exp, log, pow, 1/x, sqrt and
arithmetic operators to find expressions that map index i ∈ [0, 255]
to weight magnitude |w|.

Runs on the final (v20) checkpoint — the most converged weight distribution.
Prints the Pareto front (complexity vs. loss) and compares the best
discovered formula against the 3-param baseline G(i) = A·(exp(B·i^C)−1).
"""
import os
import sys
import warnings
import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.view.weight_distribution import load_model
from src.utils.metrics import flatten_weights
from src.view.figure import Figure

ARTIFACT_DIR = "artifacts"
MODEL_NAME = "model-87qv0k7c"
CHECKPOINT_VERSION = 20
OUTPUT_PDF = "symbolic_regression.pdf"
N_ITERATIONS = 60
MAX_SIZE = 25  # max expression complexity (nodes)


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


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ckpt = os.path.join(ARTIFACT_DIR, f"{MODEL_NAME}:v{CHECKPOINT_VERSION}", "model.ckpt")
    print(f"Loading {ckpt} ...")
    model = load_model(ckpt)
    weights = flatten_weights(model)
    idx, targets = _quantile_targets(weights)

    # Baseline fit for comparison
    base_pred, base_r2, base_label = _baseline(idx, targets)
    base_mse = _reconstruction_mse(weights, base_pred.clip(min=0))
    print(f"\nBaseline {base_label}")
    print(f"  R²={base_r2:.4f}  reconstruction MSE={base_mse:.4e}")

    # PySR search
    print(f"\nRunning PySR (niterations={N_ITERATIONS}, maxsize={MAX_SIZE}) ...")
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

    print("\n── Pareto front ──────────────────────────────────────────────")
    print(sr)

    # Best equation (lowest loss)
    best_eq = sr.sympy()
    best_pred = sr.predict(X)
    best_r2 = _r2(targets, best_pred)
    best_mse = _reconstruction_mse(weights, best_pred.clip(min=0))
    print(f"\nBest expression: {best_eq}")
    print(f"  R²={best_r2:.4f}  reconstruction MSE={best_mse:.4e}")

    # ── plots ─────────────────────────────────────────────────────────────
    equations = sr.equations_
    complexities = equations["complexity"].tolist()
    losses = equations["loss"].tolist()

    with PdfPages(OUTPUT_PDF) as pdf:
        # Page 1: Pareto front (complexity vs loss)
        fig = Figure(
            title="Symbolic regression Pareto front",
            ncols=1,
            nrows=1,
            figSize=(10, 6),
        )
        fig.plot(
            {"complexity vs loss": (complexities, losses)},
            exp_name="Pareto front  (lower-left = better)",
            x_label="expression complexity",
            y_label="MSE loss",
            symbol="o-",
            legend=False,
        )
        fig.save(pdf=pdf)

        # Page 2: best SR expression vs baseline vs empirical quantiles
        fig2 = Figure(
            title=f"Best symbolic expression vs baseline  (v{CHECKPOINT_VERSION})",
            ncols=1,
            nrows=1,
            figSize=(12, 6),
        )
        fig2.plot(
            {
                "empirical quantiles": (idx, targets),
                f"baseline  R²={base_r2:.4f}": (idx, base_pred),
                f"SR best  R²={best_r2:.4f}": (idx, best_pred),
            },
            exp_name=f"SR: {best_eq}",
            x_label="index i",
            y_label="|w|",
        )
        fig2.save(pdf=pdf)

        # Page 3: residuals comparison
        fig3 = Figure(
            title="Residuals: baseline vs best SR expression",
            ncols=2,
            nrows=1,
            figSize=(14, 5),
        )
        fig3.plot(
            {"baseline residuals": (idx, targets - base_pred)},
            exp_name=f"Baseline residuals  (R²={base_r2:.4f})",
            x_label="index i",
            y_label="residual",
            legend=False,
        )
        fig3.plot(
            {"SR residuals": (idx, targets - best_pred)},
            exp_name=f"SR residuals  (R²={best_r2:.4f})",
            x_label="index i",
            y_label="residual",
            legend=False,
        )
        fig3.save(pdf=pdf)

    print(f"\nSaved to {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
