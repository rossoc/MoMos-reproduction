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
        "L2": "metrics/weight_l2",
    }
    for name, key in metrics.items():
        fig = Figure(f"{name} vs Validation Loss", ncols=2, nrows=2)
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
                fig.save(pdf=pdf)

    if show:
        for fig in figures:
            fig.show()


class Report:
    def __init__(self, output_dir="assets"):
        self.figures = []
        self.output_dir = dir

    def new_figure(self, title=None, figSize=(10, 7), ncols=1, nrows=1, fontsize=13):
        fig = Figure(title, figSize, ncols, nrows, fontsize)
        self.figures.append(fig)
        return fig

    def save(self, filename):
        with PdfPages(self.output_dir + filename) as pdf:
            for fig in self.figures:
                fig.save(pdf=pdf)

    def metrics_vs_accuracy(self, run_data, metrics, show=False):
        for m, t in metrics.items():
            fig = self.new_figure(t + " vs Validation Loss", nrows=2, ncols=2)

            for sub_t, data in run_data.items():
                d = (
                    (range(len(data[m][0])), data[m][0], data[m][1]),
                    (
                        range(len(data["val/loss"][0])),
                        data["val/loss"][0],
                        data["val/loss"][1],
                    ),
                )
                fig.plot_twinx_with_var(d, sub_t, y1_label=t)
                if show:
                    fig.show()

    def training_overview(self, run_data, metrics, show=False):
        fig = self.new_figure("Training Overview", nrows=2, ncols=2)
        tr_overview = {
            "val/acc": "Validation Accuracy",
            "val/loss": "Validation Loss",
            "train/acc": "Training Accuracy",
            "train/loss": "Training Loss",
        }

        for m, t in tr_overview.items():
            data = {}
            for k, v in run_data.items():
                data[k] = (range(len(v[m][0])), v[m][0], v[m][1])
            print(data)
            fig.plot_with_var(data, "", y_label=t)

        if show:
            fig.show()

    def append_figures(self, figures):
        self.figures += figures
