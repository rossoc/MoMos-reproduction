# %% [markdown]
#
# %%
import pandas as pd
import wandb
from src.view import Report, extract_columns, merge_dfs
import ast
import numpy as np
from collections import defaultdict
from tqdm import tqdm

project = "momos2d-hierarchical"  # "momos-reproduction"
api = wandb.Api()
# %%

# Project is specified by <entity/project-name>
runs = api.runs(f"danesinoo-university-of-copenhagen/{project}")

summary_list, config_list, name_list = [], [], []
for run in tqdm(runs):
    summary_list.append(run.summary._json_dict)
    config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

    name_list.append(run.name)

runs_df = pd.DataFrame(
    {"summary": summary_list, "config": config_list, "name": name_list}
)

# %%
runs_df.to_csv(f"{project}.csv")
runs_df = pd.read_csv(f"{project}.csv")


# %%
report = Report()
# %%
momos2d_runs = []
for i, r in enumerate(runs_df.iterrows()):
    try:
        config = ast.literal_eval(r[1]["config"])  # type: ignore
    except Exception as _:
        config = r[1]["config"]

    try:
        momos2d_runs.append(r[1])
    except Exception as e:
        print(e)
# %%
momos2d_data = []
for r in momos2d_runs:
    try:
        summary = ast.literal_eval(r["summary"])
        config = ast.literal_eval(r["config"])
    except Exception as _:
        summary = r["summary"]
        config = r["config"]
    name = f"rows: {config['quantization']['secondary']['rows']}"
    name += f" cols: {config['quantization']['secondary']['cols']}"
    run_name = r["name"]
    bits_ratio = config["bits/ratio"]
    ras = 1 / bits_ratio
    momos2d_data.append(
        {
            "name": name,
            "run_name": r["name"],
            "val_acc": summary["val/acc"],
            "val_loss": summary["val/loss"],
            "train_acc": summary["train/acc"],
            "train_loss": summary["train/loss"],
            "test_acc": summary["test/acc"],
            "rows": config["quantization"]["secondary"]["rows"],
            "cols": config["quantization"]["secondary"]["cols"],
            "capacity": config["quantization"]["secondary"]["capacity"],
            "ras": ras,
        }
    )
# %%
momos2d_df = pd.DataFrame(momos2d_data)
stats = momos2d_df.groupby("name")[
    ["val_acc", "val_loss", "train_acc", "train_loss", "test_acc", "test_acc"]
].agg(["mean", "std"])
# %%
stats = (
    momos2d_df.groupby(["name", "capacity"])["val_acc"]
    .agg(["mean", "std"])
    .reset_index()
)
stats = stats.sort_values(["name", "capacity"])

stats_ras = (
    momos2d_df.groupby(["name", "ras"])["val_acc"].agg(["mean", "std"]).reset_index()
)
stats_ras = stats_ras.sort_values(["name", "ras"])

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

result_dict_ras = (
    stats_ras.groupby("name")
    .apply(
        lambda x: (
            np.array(x["ras"].tolist()),
            np.array(x["mean"].tolist()),
            np.array(x["std"].tolist()),
        )
    )
    .to_dict()
)

# %%
fig = report.new_figure()
fig.plot_with_var(
    result_dict,
    "",
    symbol="o-",
    x_label="Capacity",
    y_label="Validation Accuracy",
    logx=True,
)
# fig.show()
# %%
fig_ras = report.new_figure()
fig_ras.plot_with_var(
    result_dict_ras,
    "",
    symbol="o-",
    x_label="RAS",
    y_label="Validation Accuracy",
)
# fig_ras.show()
# %%
hyperparams = ["rows", "cols", "capacity"]
grouped_swapping = (
    momos2d_df.groupby(hyperparams)["val_acc"].agg(["mean", "std"]).reset_index()
)

best_params = pd.DataFrame(
    {
        "rows": [8, 8, 32, 32],
        "cols": [4, 16, 32, 4],
        "capacity": [0.0166, 0.00415, 0.00415, 0.0166],
    }
)


grouped_swapping.sort_values("mean", ascending=False)[:4][hyperparams]
detail_run_df = momos2d_df.merge(best_params, on=hyperparams, how="inner")  # type: ignore
# %%
detail_runs = [
    api.runs(
        f"danesinoo-university-of-copenhagen/{project}",
        filters={"display_name": run["run_name"]},
    )[0]
    for run in tqdm(detail_run_df.to_dict("records"))
]
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
grouped_runs = detail_run_df.groupby(hyperparams)["run_name"].apply(list).reset_index()
# %%
momos2d_best_runs_data = []
for run in tqdm(detail_runs):
    history = run.history(samples=500)
    assert type(history) is pd.DataFrame

    if history.empty:
        continue

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
runs_summary = {}
for r in grouped_runs.to_dict("records"):
    metrics = defaultdict(list)
    for run in momos2d_best_runs_data:
        if run["name"] in r["run_name"]:
            for metric in MOMOS2D_METRICS:
                if metric not in metrics:
                    metrics[metric] = []
                values = np.array(run["metrics"][metric])[:-1]
                metrics[metric] += [values]

    result = {}
    for k, v in metrics.items():
        max_len = max(len(run) for run in v)
        v = [np.append(run, [np.nan] * (max_len - len(run))) for run in v]
        mean = np.nanmean(v, axis=0)
        std = np.nanstd(v, axis=0)
        result[k] = (mean, std)

    runs_summary[f"rows: {r['rows']} cols: {r['cols']} cap: {r['capacity']}"] = result
# %%
tr_overview = {
    "val/acc": "Validation Accuracy",
    "val/loss": "Validation Loss",
    "train/acc": "Training Accuracy",
    "train/loss": "Training Loss",
}
report.training_overview(runs_summary, tr_overview, show=True)
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

report.metrics_vs_accuracy(runs_summary, metrics_overview, True)
# %%
report.save("momos_hierarchical_overview.pdf")
