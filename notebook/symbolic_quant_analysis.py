import os
import sys
import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
import warnings
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.view.weight_distribution import load_model
from src.utils.metrics import flatten_weights
from src.view.figure import Figure

ARTIFACT_DIR = "artifacts"
MODEL_NAME = "model-87qv0k7c"
N_VERSIONS = 20
OUTPUT_PDF = "symbolic_quant_analysis.pdf"


def _G(i, A, B, C):
    # C > 1 bends the index curve upward, spacing codebook entries more densely
    # in the high-magnitude tail — preserving accuracy for significant large weights.
    return A * (np.exp(B * np.power(i, C)) - 1)


def fit_version(checkpoint_path):
    model = load_model(checkpoint_path)
    weights = flatten_weights(model)

    magnitudes = np.sort(np.abs(weights))
    idx = np.arange(256, dtype=float)
    percentiles = np.linspace(0, 100, 256)
    targets = np.percentile(magnitudes, percentiles)

    # C=1 recovers the original 2-param form; optimizer will push it above 1
    # to better track the sparse but significant high-value weights.
    p0 = [max(targets[-1], 1e-9) / (np.e - 1), 1.0 / 255, 1.0]
    bounds = ([0, 0, 0.1], [np.inf, np.inf, 10.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        try:
            (A, B, C), _ = curve_fit(
                _G, idx, targets, p0=p0, bounds=bounds, maxfev=20000
            )
        except RuntimeError:
            return None

    predicted = _G(idx, A, B, C)
    ss_res = np.sum((targets - predicted) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse_fit = float(np.sqrt(np.mean((targets - predicted) ** 2)))

    codebook = _G(np.arange(256, dtype=float), A, B, C)
    signs = np.sign(weights)
    mags = np.abs(weights)
    enc_idx = np.argmin(np.abs(mags[:, None] - codebook[None, :]), axis=1)
    reconstructed = signs * codebook[enc_idx]
    mse = float(np.mean((weights - reconstructed) ** 2))

    return {
        "A": float(A),
        "B": float(B),
        "C": float(C),
        "r2": float(r2),
        "rmse_fit": rmse_fit,
        "mse": mse,
        "idx": idx,
        "targets": targets,
        "predicted": predicted,
        "residuals": targets - predicted,
    }


def _plot_version_page(pdf, version, result):
    fig = Figure(
        title=f"{MODEL_NAME} v{version}  —  A={result['A']:.3e}  B={result['B']:.3e}  C={result['C']:.3f}  R²={result['r2']:.4f}  MSE={result['mse']:.3e}",
        ncols=2,
        nrows=1,
        figSize=(14, 5),
    )
    fig.plot(
        {
            "empirical quantiles": (result["idx"], result["targets"]),
            "G(i) fit": (result["idx"], result["predicted"]),
        },
        exp_name="Magnitude distribution vs G(i)",
        x_label="index i",
        y_label="|w|",
    )
    fig.plot(
        {"residuals": (result["idx"], result["residuals"])},
        exp_name="Fit residuals (targets − G(i))",
        x_label="index i",
        y_label="residual",
        legend=False,
    )
    fig.save(pdf=pdf)


def _plot_summary_page(pdf, versions, results):
    versions_arr = np.array(versions, dtype=float)
    A_arr = np.array([r["A"] for r in results])
    B_arr = np.array([r["B"] for r in results])
    C_arr = np.array([r["C"] for r in results])
    r2_arr = np.array([r["r2"] for r in results])
    mse_arr = np.array([r["mse"] for r in results])

    fig = Figure(
        title="G(i) = A·(exp(B·i^C)−1) parameters across training checkpoints",
        ncols=3,
        nrows=2,
        figSize=(18, 10),
    )
    fig.plot(
        {"A": (versions_arr, A_arr)},
        exp_name="Scale parameter A",
        x_label="checkpoint version",
        y_label="A",
        legend=False,
    )
    fig.plot(
        {"B": (versions_arr, B_arr)},
        exp_name="Growth parameter B",
        x_label="checkpoint version",
        y_label="B",
        legend=False,
    )
    fig.plot(
        {"C": (versions_arr, C_arr)},
        exp_name="Shape exponent C  (>1 = tail resolution)",
        x_label="checkpoint version",
        y_label="C",
        legend=False,
    )
    fig.plot(
        {"R²": (versions_arr, r2_arr)},
        exp_name="Fit quality R²",
        x_label="checkpoint version",
        y_label="R²",
        legend=False,
    )
    fig.plot(
        {"MSE": (versions_arr, mse_arr)},
        exp_name="Reconstruction MSE",
        x_label="checkpoint version",
        y_label="MSE",
        legend=False,
    )
    fig.save(pdf=pdf)


def main():
    all_versions = []
    all_results = []

    with PdfPages(OUTPUT_PDF) as pdf:
        for v in range(1, N_VERSIONS + 1):
            ckpt = os.path.join(ARTIFACT_DIR, f"{MODEL_NAME}:v{v}", "model.ckpt")
            if not os.path.exists(ckpt):
                print(f"  v{v}: checkpoint not found, skipping")
                continue

            result = fit_version(ckpt)
            if result is None:
                print(f"  v{v}: curve_fit failed, skipping")
                continue

            print(
                f"  v{v}: A={result['A']:.4e}  B={result['B']:.4e}  C={result['C']:.4f}"
                f"  R²={result['r2']:.4f}  MSE={result['mse']:.4e}"
            )
            _plot_version_page(pdf, v, result)
            all_versions.append(v)
            all_results.append(result)

        if len(all_results) > 1:
            _plot_summary_page(pdf, all_versions, all_results)

    print(f"\nSaved to {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
