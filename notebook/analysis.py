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
from src.view import fetch_runs, Figure
import numpy as np
import pickle

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
momos_data = {"k/m": [], "s": [], "val_acc": []}

for exp in experiments:
    if (
        exp["config"].get("quantization", None) is not None
        and exp["config"]["quantization"]["method"] == "momos"
    ):
        q_config = exp["config"]["quantization"]
        exp["val/acc"] = float(np.max(exp["metrics"]["val/acc"]))

        exp["s"] = q_config["s"]
        exp["capacity"] = q_config["capacity"]

        momos_data["k/m"] += [float(q_config["capacity"])]
        momos_data["s"] += [q_config["s"]]
        momos_data["val_acc"] += [float(np.max(exp["metrics"]["val/acc"]))]

        momos_runs.append(exp)

for d in momos_data.values():
    d = d[1:]
grouped_momos_data = {}
for r in momos_runs:
    q_config = r["config"]["quantization"]
    s_value = q_config["s"]
    k_m_value = float(q_config["capacity"])
    val_acc_value = float(np.max(r["metrics"]["val/acc"]))

    if s_value not in grouped_momos_data:
        grouped_momos_data[s_value] = {"k/m_s": [], "val_acc": []}
    grouped_momos_data[s_value]["k/m_s"].append(k_m_value)
    grouped_momos_data[s_value]["val_acc"].append(val_acc_value)

momos_runs.sort(key=lambda x: x["val/acc"], reverse=True)
# %%
fig = Figure()
fig.plot(
    data={
        exp["name"]: exp["metrics"][["epoch", "val/acc"]] for exp in experiments[-5:]
    },
    exp_name="Validation Loss",
)
fig.show()
# %%
fig = Figure(title="Experiments", ncols=2, nrows=2)
fig.plot(
    data={
        exp["name"]: exp["metrics"][["epoch", "val/acc"]] for exp in experiments[-5:]
    },
    exp_name="Validation Accuracy",
)
fig.plot(
    data={
        exp["name"]: exp["metrics"][["epoch", "train/acc"]] for exp in experiments[-5:]
    },
    exp_name="Training Accuracy",
)
fig.plot(
    data={
        exp["name"]: exp["metrics"][["epoch", "val/loss"]] for exp in experiments[-5:]
    },
    exp_name="Validation Loss",
)
fig.plot(
    data={
        exp["name"]: exp["metrics"][["epoch", "train/loss"]] for exp in experiments[-5:]
    },
    exp_name="Training Loss",
)
fig.show()
# %%
fig = Figure()
fig.plot3D(
    (momos_data["k/m"], momos_data["s"], momos_data["val_acc"]),
    x_label="Capacity",
    y_label="Block size",
    z_label="Validation accuracy",
    exp_name="MoMos: Accuracy vs. block size and capacity",
    cmap="magma",
    logy=2,
)
fig.show()
fig.save("acc_vs_s_vs_capacity")
# %%
fig_s_grouped = Figure(title="MoMos: Capacity vs. Accuracy grouped by S")
fig_s_grouped.plot(
    {f"S={s}": (d["k/m_s"], d["val_acc"]) for s, d in grouped_momos_data.items()},
    exp_name=None,  # type: ignore
)
fig_s_grouped.plot_index -= 1
fig_s_grouped.plot(
    {"baseline": ([0, 0.3], [np.max(experiments[0]["metrics"]["val/acc"])] * 2)},
    x_label="Capacity",
    y_label="Validation Accuracy",
    exp_name="MoMos: Grouped by Block Size (s)",  # Overall plot experiment name/description
    symbol="--",
    colors=["black"],
)
fig_s_grouped.show()
# fig_s_grouped.save("acc_vs_capacity_grouped_by_s")
# %%

fig = Figure("Distortion vs Acc", ncols=2, nrows=2)
for i in range(4):
    fig.plot_twinx(
        (
            momos_runs[i]["metrics"][["epoch", "quant/distortion"]],
            momos_runs[i]["metrics"][["epoch", "val/loss"]],
        ),
        f"S = {momos_runs[i]['s']}, capacity = {momos_runs[i]['capacity']}",
        y1_label="Distortion",
        y2_label="Val. Loss",
    )
fig.show()
# %%
