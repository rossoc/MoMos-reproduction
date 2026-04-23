# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3.12.3.final.0 (momos-reproduction; uv)
#     language: python
#     name: Python 3.12.3.final.0 (momos-reproduction; uv)
# ---

# %% [markdown]
# # Data Exploration
# ## Data Import

# %%
from src.view import fetch_runs
from src.view.report import report
from src.view.weight_distribution import plot_weights

import numpy as np
import pickle
import wandb
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

# %%
# Define cache file path
CACHE_FILE = "outputs/experiments_cache.pkl"

# Try to load from cache first
if __import__("os").path.exists(CACHE_FILE):
    with open(CACHE_FILE, "rb") as f:
        experiments, _, _, _ = pickle.load(f)
    print(f"Loaded {len(experiments)} experiments from cache: {CACHE_FILE}")
else:
    # Fetch and save to cache
    experiments, _, _, _ = fetch_runs()
    with open(CACHE_FILE, "wb") as f:
        pickle.dump((experiments, None, None, None), f)
    print(f"Saved {len(experiments)} experiments to cache: {CACHE_FILE}")
# %%
momos_runs = []
qat_runs = []
baseline_runs = []
momos_data = {"capacity": [], "s": [], "val_acc": []}

for exp in experiments:
    exp["val/acc"] = float(np.max(exp["metrics"]["val/acc"]))
    exp["run_name"] = exp["name"]
    if (
        exp["config"].get("quantization", None) is not None
        and exp["config"]["quantization"]["method"] == "momos"
    ):
        q_config = exp["config"]["quantization"]

        exp["s"] = q_config["s"]
        exp["capacity"] = q_config["capacity"]

        momos_data["capacity"] += [float(q_config["capacity"])]
        momos_data["s"] += [q_config["s"]]
        momos_data["val_acc"] += [float(np.max(exp["metrics"]["val/acc"]))]

        exp["name"] = f"momos s={exp['s']} cap={exp['capacity']}"

        momos_runs.append(exp)
    elif (
        exp["config"].get("quantization", None) is not None
        and exp["config"]["quantization"]["method"] == "qat"
    ):
        exp["name"] = "qat"
        qat_runs.append(exp)
    elif exp["config"].get("quantization", None) is None:
        exp["name"] = "baseline"
        baseline_runs.append(exp)

for d in momos_data.values():
    d = d[1:]
grouped_momos_data = {}
for r in momos_runs:
    q_config = r["config"]["quantization"]
    s_value = q_config["s"]
    k_m_value = float(q_config["capacity"])
    val_acc_value = float(np.max(r["metrics"]["val/acc"]))

    if s_value not in grouped_momos_data:
        grouped_momos_data[s_value] = {"capacity": [], "val_acc": []}
    grouped_momos_data[s_value]["capacity"].append(k_m_value)
    grouped_momos_data[s_value]["val_acc"].append(val_acc_value)

momos_runs.sort(key=lambda x: x["val/acc"], reverse=True)
qat_runs.sort(key=lambda x: x["val/acc"], reverse=True)
baseline_runs.sort(key=lambda x: x["val/acc"], reverse=True)
experiments.sort(key=lambda x: x["val/acc"], reverse=True)
# %%
report(
    "Experiment1",
    [baseline_runs[0]] + [qat_runs[0]] + momos_runs,
    momos_data,
    grouped_momos_data,
    momos_runs,
    show=False,
)


# %%
for r in momos_runs[:4]:
    print(r["run_name"])
# %%
best_runs_artifacts = [
    "model-shlxpkpa:v0",  # 23
    "model-ab1nlxax:v0",  # 21
    "model-upxb9glw:v0",  # 20
    "model-x63vszuv:v0",  # 22
    "model-26q8yc0n:v0",  # 53
]
api = wandb.Api()
for r in best_runs_artifacts[-1:]:
    artifact = api.artifact(
        f"danesinoo-university-of-copenhagen/momos-reproduction/{r}",
        type="model",
    )
    artifact_dir = artifact.download()
# %%
data = [
    ("artifacts/model-shlxpkpa:v0/model.ckpt", 2, 0.3),  # run 23
    ("artifacts/model-ab1nlxax:v0/model.ckpt", 2, 0.1),  # run 21
    ("artifacts/model-upxb9glw:v0/model.ckpt", 2, 0.05),  # run 20
    ("artifacts/model-x63vszuv:v0/model.ckpt", 2, 0.2),  # run 22
]
# %%
figures = []
for d in data:
    figures += plot_weights(d)

with PdfPages("weight_analysis_2.pdf") as pdf:
    for fig in figures:
        fig.save(pdf=pdf)


# %%
plt.close()
# %%
folder = "outputs/cifar10_mlp/0_fantastic-peridot-dingo-of-debate"
one_run_many_epochs = [
    (
        f"{folder}/init.ckpt",
        2,
        "0.05 initialization",
    ),
    (
        f"{folder}/epoch-epoch=09.ckpt",
        2,
        "0.05 epoch=09",
    ),
] + [
    (
        f"{folder}/epoch-epoch={epoch}.ckpt",
        2,
        "0.05 epoch={epoch}",
    )
    for epoch in range(19, 100, 10)
]
# %%
figures = []
for d in one_run_many_epochs:
    figures += plot_weights(d)
with PdfPages("weight_analysis_per_epoch.pdf") as pdf:
    for fig in figures:
        fig.save(pdf=pdf)
# %%
run_s128 = [("artifacts/model-26q8yc0n:v0/model.ckpt", 128, 0.3)][0]
figures = plot_weights(run_s128)
with PdfPages("weight_analysis_s=128.pdf") as pdf:
    for fig in figures:
        fig.save(pdf=pdf)
