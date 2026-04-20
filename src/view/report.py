from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from .figure import Figure


def report(filename, experiments, momos_data, group_by, momos_runs, show=True):
    # %% Training overview
    metrics = {
        "Validation Accuracy": "val/acc",
        "Training Accuracy": "train/acc",
        "Validation Loss": "val/loss",
        "Training Loss": "train/loss",
    }

    runs = experiments[:5]

    fig_training = Figure(title="Training Overview", ncols=2, nrows=2)
    for name, metric in metrics.items():
        _ = fig_training.plot(
            data={r["name"]: r["metrics"][["epoch", metric]] for r in runs},
            exp_name=name,
        )

    # %% 3D group by
    fig_3D = Figure()
    fig_3D.plot3D(
        (momos_data["capacity"], momos_data["s"], momos_data["val_acc"]),
        x_label="Capacity",
        y_label="Block size",
        z_label="Validation accuracy",
        exp_name="MoMos: Accuracy vs. block size and capacity",
        cmap="magma",
        logy=2,
    )

    # %% Group by
    fig_group_by = Figure(title="MoMos: Capacity vs. Accuracy grouped by block Size")
    fig_group_by.plot(
        {f"S={s}": (d["capacity"], d["val_acc"]) for s, d in group_by.items()},
        exp_name=None,  # type: ignore
        symbol="o-",
    )
    fig_group_by.plot_index -= 1
    fig_group_by.plot(
        {"baseline": ([0, 0.3], [np.max(experiments[0]["metrics"]["val/acc"])] * 2)},
        x_label="Capacity",
        y_label="Validation Accuracy",
        exp_name="",
        symbol="--",
        colors=["black"],
    )

    figures = [
        fig_training,
        fig_3D,
        fig_group_by,
    ]
    metrics = {
        "Distortion": "quant/distortion",
        "BZ2 Compression Rate": "metrics/bz2_compression_rate",
        "LZMA Compression Rate": "metrics/lzma_compression_rate",
        "BDM Complexity": "metrics/bdm_complexity",
        "GZip Compression Rate": "metrics/gzip_compression_rate",
    }
    for name, key in metrics.items():
        fig = Figure(f"{name} vs Accuracy", ncols=2, nrows=2)
        for i in range(4):
            _ = fig.plot_twinx(
                (
                    momos_runs[i]["metrics"][["epoch", key]],
                    momos_runs[i]["metrics"][["epoch", "val/loss"]],
                ),
                f"S = {momos_runs[i]['s']}, capacity = {momos_runs[i]['capacity']}",
                x_label="Epochs",
                y1_label=name,
                y2_label="Val. Loss",
            )
        figures.append(fig)

    if filename is not None:
        with PdfPages(filename + ".pdf") as pdf:
            for fig in figures:
                fig.save("", pdf)

    if show:
        for fig in figures:
            fig.show()
            

def _compute_rac(run):
    s = run["config"]["s"]
    k = run["config"]["k"]
    q = run["config"]["q"]

