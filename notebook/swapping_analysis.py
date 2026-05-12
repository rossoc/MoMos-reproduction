# %% [markdown]
# %%
import pandas as pd
import wandb
import ast
import numpy as np
from src.view import extract_columns, merge_dfs, Report
from tqdm import tqdm
# %%

api = wandb.Api()
prj = "momos-collapse"


# %%
runs = api.runs(f"danesinoo-university-of-copenhagen/{prj}")
summary_list, config_list, name_list = [], [], []
for run in tqdm(runs):
    summary_list.append(run.summary._json_dict)
    config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})
    name_list.append(run.name)

runs_df = pd.DataFrame(
    {"summary": summary_list, "config": config_list, "name": name_list}
)

runs_df.to_csv("{prj}.csv")
# %%
runs_df = pd.read_csv("{prj}.csv")
report = Report()
# %%

percentile_runs = []
for r in runs_df.iterrows():
    try:
        config = ast.literal_eval(r[1]["config"])
    except Exception as _:
        config = r[1]["config"]

    if (
        config.get("quantization", None)
        and config["quantization"].get("swapping_probability", None)
        and config.get("epochs", None) == 400
    ):
        run = r[1]
        quant_cfg = config["quantization"]

        try:
            summary = ast.literal_eval(r[1]["summary"])
        except Exception as _:
            summary = r[1]["summary"]

        if not summary.get("val/acc", None):
            continue

        p_from = int(quant_cfg["from_percentile"][0] * 100)
        p_to = int(quant_cfg["to_percentile"][0] * 100)
        p_swap = float(quant_cfg["swapping_probability"])

        percentile_runs.append(
            {
                "name1": f"Replace with {p_to}th",
                "name2": f"Replace {p_from}th with {p_to}th",
                "name3": f"Replace {p_from}th",
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
fig = report.new_figure()
fig.plot_with_var(
    from_percentile_data,
    "Percentile Ablations (avg. on swapping probability)",
    x_label="Percentile changed",
    y_label="Validation accuracy",
    symbol="o-",
)
fig.show()
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
fig = report.new_figure()
fig.plot_with_var(
    p_swap_data,
    "",
    x_label="Swapping probability",
    y_label="Validation accuracy",
    symbol="o-",
)
fig.show()

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
fig = report.new_figure()
fig.plot_with_var(
    p_from_data,
    "",
    x_label="Changing with percentile",
    y_label="Validation accuracy",
    symbol="o-",
)
fig.show()
# %%
hyperparams = ["from_percentile", "to_percentile", "p_swap"]
grouped_swapping = (
    percentile_df.groupby(hyperparams)["val_acc"].agg(["mean", "std"]).reset_index()
)

best_params = grouped_swapping.sort_values("mean", ascending=False)[:4][hyperparams]
best_run_df = percentile_df.merge(best_params, on=hyperparams)
# %%
best_runs = []
for name in tqdm(best_run_df["run_name"].tolist()):
    run = api.runs(
        f"danesinoo-university-of-copenhagen/{prj}",
        filters={"display_name": name},
    )
    best_runs.append(run[0])
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
    "quant/num_changed_weights",
    "val/loss",
    "val/acc",
    "train/loss",
    "train/acc",
]


# %%
best_runs_data = []

for run in tqdm(best_runs):
    history = run.history(samples=50000)
    assert type(history) is pd.DataFrame

    try:
        metrics = []
        metrics += [history[["epoch"]].dropna().drop_duplicates()]
        metrics += extract_columns(history, MOMOS2D_METRICS)

    except Exception as e:
        print(run.name)
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
best_runs_data
# %%
metrics = {}
best_runs_summary = {}
for r in grouped_runs.iterrows():
    for run in best_runs_data:
        if run["name"] in r[1]["run_name"]:
            for metric in MOMOS2D_METRICS:
                if metric not in metrics:
                    metrics[metric] = []
                metrics[metric] += [np.array(run["metrics"][metric].tolist())[:-1]]

    result = {}
    for k, v in metrics.items():
        max_len = max(len(run) for run in v)
        v = [np.append(run, [np.nan] * (max_len - len(run))) for run in v]
        mean = np.nanmean(v, axis=0)
        std = np.nanstd(v, axis=0)
        result[k] = (mean, std)

    best_runs_summary[
        f"Switch {r[1]['from_percentile']}th with {r[1]['to_percentile']}th w.p. {r[1]['p_swap']}"
    ] = result
# %%
tr_overview = {
    "val/acc": "Validation Accuracy",
    "val/loss": "Validation Loss",
    "train/acc": "Training Accuracy",
    "train/loss": "Training Loss",
}

report.training_overview(best_runs_summary, tr_overview, show=True)

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

report.metrics_vs_accuracy(best_runs_summary, metrics_overview, True)
# %%
report.save("percentile_ablations.pdf")
