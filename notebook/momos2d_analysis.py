# %% [markdown]
#
# %%
import pandas as pd
import wandb
from src.view import Figure
import ast
import numpy as np
import matplotlib.pyplot as plt
from src.view.fetch_log import extract_columns, merge_dfs
from matplotlib.backends.backend_pdf import PdfPages

# %%
runs_df = pd.read_csv("project.csv")
figures = []
# %%
momos2d_runs = []
for r in runs_df.iterrows():
    config = ast.literal_eval(r[1]["config"])
    if (
        config.get("quantization", None)
        and config["quantization"].get("method", None) == "momos2d"
    ):
        momos2d_runs.append(r[1])
# %%
momos2d_data = []
for r in momos2d_runs:
    summary = ast.literal_eval(r["summary"])
    config = ast.literal_eval(r["config"])
    name = "momos2d"
    name += f" rows: {config['quantization']['rows']}"
    name += f" cols: {config['quantization']['cols']}"
    run_name = r["name"]
    momos2d_data.append(
        {
            "name": name,
            "run_name": r["name"],
            "val_acc": summary["val/acc"],
            "val_loss": summary["val/loss"],
            "train_acc": summary["train/acc"],
            "train_loss": summary["train/loss"],
            "rows": config["quantization"]["rows"],
            "cols": config["quantization"]["cols"],
            "capacity": config["quantization"]["capacity"],
        }
    )
# %%
momos2d_df = pd.DataFrame(momos2d_data)
stats = momos2d_df.groupby("name")[
    ["val_acc", "val_loss", "train_acc", "train_loss"]
].agg(["mean", "std"])
# %%
stats.columns
# %%
stats = (
    momos2d_df.groupby(["name", "capacity"])["val_acc"]
    .agg(["mean", "std"])
    .reset_index()
)

# 2. Sort by capacity to ensure the lists/arrays are ordered correctly for plotting
stats = stats.sort_values(["name", "capacity"])

# 3. Create the dictionary mapping name to ([capacities], [means], [stds])
result_dict = (
    stats.groupby("name")
    .apply(
        lambda x: (
            np.array(x["capacity"].tolist()),
            np.array(x["mean"].tolist()),
            np.array(x["std"].tolist()),
        )
    )
    .to_dict()
)
# %%
fig = Figure()
fig.plot_with_var(
    result_dict, "", symbol="o-", x_label="Capacity", y_label="Validation Accuracy"
)
fig.show()
figures += [fig]
# %%

plt.close()
# %%

best_params = [(2, 1, 0.1), (2, 1, 0.05), (2, 1, 0.01), (2, 2, 0.1)]
momos2d_best = []
for params in best_params:
    momos2d_best.append(
        momos2d_df[
            (momos2d_df["rows"] == params[0])
            & (momos2d_df["cols"] == params[1])
            & (momos2d_df["capacity"] == params[2])
        ]
    )
momos2d_best = pd.concat(momos2d_best, ignore_index=True)


# %%
# %%
api = wandb.Api()
runs = api.runs("danesinoo-university-of-copenhagen/momos-reproduction")

momos2d_best_runs = []
for r in runs:
    if r.name in momos2d_best["run_name"].tolist():
        momos2d_best_runs.append(r)
# %%

MOMOS2D_METRICS = [
    "metrics/bdm_complexity",
    "metrics/gzip_compression_rate",
    "metrics/bz2_compression_rate",
    "metrics/lzma_compression_rate",
    "metrics/weight_l2",
    "metrics/sparsity",
    "quant/distortion",
    "quant/num_changed_weights",
    "val/loss",
    "val/acc",
    "train/loss",
    "train/acc",
]


# %%
momos2d_best_runs_data = []

for run in momos2d_best_runs:
    history = run.history(samples=500)
    assert type(history) is pd.DataFrame

    try:
        metrics = []
        metrics += [history[["epoch"]].dropna().drop_duplicates()]
        metrics += extract_columns(history, MOMOS2D_METRICS)

    except Exception as e:
        print(history.columns)
        print(e)
        break

    metrics = merge_dfs(metrics)
    momos2d_best_runs_data.append(
        {
            "name": run.name,
            "metrics": metrics,
            "config": run.config,
        }
    )
# %%
for r in momos2d_best_runs_data:
    print(r["name"])


# %%
grouped_runs = (
    momos2d_df.groupby(["rows", "cols", "capacity"])["run_name"]
    .apply(list)
    .reset_index()
)


# %%

best_runs_summary = {}
for params in best_params:
    metrics = {}

    for n in grouped_runs[
        (grouped_runs["rows"] == params[0])
        & (grouped_runs["cols"] == params[1])
        & (grouped_runs["capacity"] == params[2])
    ]["run_name"]:
        for r in momos2d_best_runs_data:
            if r["name"] in n:
                for metric in MOMOS2D_METRICS:
                    if metric not in metrics:
                        metrics[metric] = []
                    metrics[metric] += [np.array(r["metrics"][metric].tolist())]

    result = {}
    for k, v in metrics.items():
        mean = np.mean(v, axis=0)[:-1]
        std = np.std(v, axis=0)[:-1]
        result[k] = (mean, std)

    best_runs_summary[
        f"momos2d rows={params[0]} cols={params[1]} capacity={params[2]}"
    ] = result
# %%

fig = Figure("Training Overview", nrows=2, ncols=2)
tr_overview = {
    "val/acc": "Validation Accuracy",
    "val/loss": "Validation Loss",
    "train/acc": "Training Accuracy",
    "train/loss": "Training Loss",
}

for m, t in tr_overview.items():
    data = {}
    for k, v in best_runs_summary.items():
        data[k] = (range(len(v[m][0])), v[m][0], v[m][1])
    print(data)
    fig.plot_with_var(data, "", y_label=t)
fig.show()
figures += [fig]


# %%
metrics_overview = {
    "metrics/bdm_complexity": "BDM Complexity",
    "metrics/gzip_compression_rate": "Gzip Compression Rate",
    "metrics/bz2_compression_rate": "BZ2 Compression Rate",
    "metrics/lzma_compression_rate": "LZMA Compression Rate",
    "metrics/weight_l2": "Weight L2",
    "metrics/sparsity": "Sparsity",
    "quant/distortion": "Quantization Distortion",
    "quant/num_changed_weights": "Number of Changed Weights",
}

for m, t in metrics_overview.items():
    fig = Figure(t + " vs Validation Loss", nrows=2, ncols=2)

    for sub_t, data in best_runs_summary.items():
        d = (
            (range(len(data[m][0])), data[m][0], data[m][1]),
            (range(len(data["val/loss"][0])), data["val/loss"][0], data["val/loss"][1]),
        )
        fig.plot_twinx_with_var(d, sub_t, y1_label=t)
        fig.show()

    figures += [fig]
# %%
with PdfPages("momos2_overview.pdf") as pdf:
    for fig in figures:
        fig.save(pdf=pdf)
# %%
