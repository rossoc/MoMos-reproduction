# %% [markdown]
# %%
import pandas as pd
import wandb
from src.view import Figure
import ast
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from src.view.fetch_log import extract_columns, merge_dfs

# %%

# api = wandb.Api()
#
# # Project is specified by <entity/project-name>
# runs = api.runs("danesinoo-university-of-copenhagen/momos-reproduction")
#
# summary_list, config_list, name_list = [], [], []
# for run in runs:
#     # .summary contains the output keys/values for metrics like accuracy.
#     #  We call ._json_dict to omit large files
#     summary_list.append(run.summary._json_dict)
#
#     # .config contains the hyperparameters.
#     #  We remove special values that start with _.
#     config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})
#
#     # .name is the human-readable name of the run.
#     name_list.append(run.name)
#
# runs_df = pd.DataFrame(
#     {"summary": summary_list, "config": config_list, "name": name_list}
# )
#
# runs_df.to_csv("project.csv")

runs_df = pd.read_csv("project.csv")
figures = []
# %%

percentile_runs = []
for r in runs_df.iterrows():
    config = ast.literal_eval(r[1]["config"])
    if config.get("quantization", None) and config["quantization"].get(
        "swapping_probability", None
    ):
        run = r[1]
        quant_cfg = config["quantization"]

        summary = ast.literal_eval(r[1]["summary"])

        if not summary.get("val/acc", None):
            continue

        p_from = int(quant_cfg["from_percentile"][0] * 100)
        p_to = int(quant_cfg["to_percentile"][0] * 100)
        p_swap = float(quant_cfg["swapping_probability"])

        percentile_runs.append(
            {
                "name1": f"Change with {p_to}th",
                "name2": f"{p_from}th <-- {p_to}th",
                "name3": f"Changing {p_from}th",
                "run_name": r[1]["name"],
                "val_acc": summary["val/acc"],
                "from_percentile": p_from,
                "to_percentile": p_to,
                "p_swap": p_swap,
            }
        )
percentile_df = pd.DataFrame(percentile_runs)
# %%
from_percentile_df = (
    percentile_df.groupby(["from_percentile", "name1"])["val_acc"]
    .agg(["mean", "std"])
    .reset_index()
)
from_percentile_data = (
    from_percentile_df.groupby("name1")
    .apply(
        lambda x: (
            np.array(x["from_percentile"].tolist()),
            np.array(x["mean"].tolist()),
            np.array(x["std"].tolist()),
        )
    )
    .to_dict()
)
# %%
fig = Figure()
fig.plot_with_var(
    from_percentile_data,
    "Percentile Ablations (avg. on swapping probability)",
    x_label="Percentile changed",
    y_label="Validation accuracy",
    symbol="o-",
)
fig.show()
figures += [fig]
# %%
p_swap_df = (
    percentile_df.groupby(["p_swap", "name2"])["val_acc"]
    .agg(["mean", "std"])
    .reset_index()
)
p_swap_data = (
    p_swap_df.groupby("name2")
    .apply(
        lambda x: (
            np.array(x["p_swap"].tolist()),
            np.array(x["mean"].tolist()),
            np.array(x["std"].tolist()),
        )
    )
    .to_dict()
)
fig = Figure()
fig.plot_with_var(
    p_swap_data,
    "",
    x_label="Swapping probability",
    y_label="Validation accuracy",
    symbol="o-",
)
fig.show()
figures += [fig]


# %%
p_from_df = (
    percentile_df.groupby(["to_percentile", "name3"])["val_acc"]
    .agg(["mean", "std"])
    .reset_index()
)
p_from_data = (
    p_from_df.groupby("name3")
    .apply(
        lambda x: (
            np.array(x["to_percentile"].tolist()),
            np.array(x["mean"].tolist()),
            np.array(x["std"].tolist()),
        )
    )
    .to_dict()
)
fig = Figure()
fig.plot_with_var(
    p_from_data,
    "",
    x_label="Changing with percentile",
    y_label="Validation accuracy",
    symbol="o-",
)
fig.show()
figures += [fig]
# %%
hyperparams = ["from_percentile", "to_percentile", "p_swap"]
grouped_swapping = (
    percentile_df.groupby(hyperparams)["val_acc"].agg(["mean", "std"]).reset_index()
)

best_params = grouped_swapping.sort_values("mean", ascending=False)[:4][hyperparams]
best_run_df = percentile_df.merge(best_params, on=hyperparams)
# %%
api = wandb.Api()
runs = api.runs("danesinoo-university-of-copenhagen/momos-reproduction")

best_runs = []
for r in runs:
    if r.name in best_run_df["run_name"].tolist():
        best_runs.append(r)


# %%
grouped_runs = best_run_df.groupby(hyperparams)["run_name"].apply(list).reset_index()
# %%
MOMOS2D_METRICS = [
    "metrics/bdm_complexity",
    "metrics/gzip_compression_rate",
    "metrics/bz2_compression_rate",
    "metrics/lzma_compression_rate",
    "metrics/weight_l2",
    "metrics/sparsity",
    "quant/distortion",
    "quant/changed_weights",
    "val/loss",
    "val/acc",
    "train/loss",
    "train/acc",
]


# %%
best_runs_data = []

for run in best_runs:
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
    best_runs_data.append(
        {
            "name": run.name,
            "metrics": metrics,
            "config": run.config,
        }
    )
# %%
grouped_runs
# %%


metrics = {}
best_runs_summary = {}
for r in grouped_runs.iterrows():
    print(r[1]["run_name"])
    for run in best_runs_data:
        if run["name"] in r[1]["run_name"]:
            for metric in MOMOS2D_METRICS:
                if metric not in metrics:
                    metrics[metric] = []
                metrics[metric] += [np.array(run["metrics"][metric].tolist())[:-1]]

    result = {}
    for k, v in metrics.items():
        mean = np.mean(v, axis=0)
        std = np.std(v, axis=0)
        result[k] = (mean, std)

    best_runs_summary[
        f"Switch {r[1]['from_percentile']}th with {r[1]['to_percentile']}th w.p. {r[1]['p_swap']}"
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
    "quant/changed_weights": "Number of Changed Weights",
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
with PdfPages("percentile_ablations.pdf") as pdf:
    for fig in figures:
        fig.save(pdf=pdf)

figures = []
# %%


# %%
