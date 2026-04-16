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

momos_config = []
for exp in experiments:
    if (
        exp["config"].get("quantization", None)
        and exp["config"]["quantization"]["method"] == "momos"
    ):
        q_config = exp["config"]["quantization"]
        momos_config.append(
            {
                "capacity": q_config["capacity"],
                "s": q_config["s"],
                "test_acc": float(exp["test_acc"]),
            }
        )
momos_config = momos_config[1:]
# %%


s_vals = [d["s"] for d in momos_config]
caps = [d["capacity"] for d in momos_config]
accs = [d["test_acc"] for d in momos_config]
data = [("Block size", s_vals), ("Capacity", caps), ("Test accuracy", accs)]

fig = Figure()
fig.plot3D(data, "MoMos: Accuracy vs. block size and capacity", cmap="magma", logx=2)
fig.show()
fig.save("acc_vs_s_vs_capacity")
# %%
