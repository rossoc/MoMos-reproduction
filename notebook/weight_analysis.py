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
from src.view.report import Report
from src.view.weight_distribution import plot_weights_2d
from tqdm import trange

import wandb
import os

# %%
api = wandb.Api()
runs = ["model-qcywyfqe"]
_config_cache = {}
for run_name in runs:
    report = Report()
    rows = cols = cap = None
    for i in trange(1, 21):
        artifact = api.artifact(
            f"danesinoo-university-of-copenhagen/momos-collapse/{run_name}:v{i}"
        )
        if run_name not in _config_cache:
            q_cfg = artifact.logged_by().config.get("quantization", {})
            _config_cache[run_name] = q_cfg
        q_cfg = _config_cache[run_name]
        rows, cap = q_cfg["s"], q_cfg["capacity"]
        cols = 1
        d = artifact.download()
        ckpt_path = os.path.join(d, "model.ckpt")
        if os.path.exists(ckpt_path):
            report.append_figures(plot_weights_2d(ckpt_path, rows, cols, i * 20))
    if rows is not None:
        report.save(f"momos2d_wa_r{rows}_c{cols}_cap{cap}.pdf")
# %%
rows = 1
cols = 2
cap = 0.001
ckpt_path = "final.pt"
report = Report()
if os.path.exists(ckpt_path):
    report.append_figures(plot_weights_2d(ckpt_path, rows, cols, 400))
if rows is not None:
    report.save(f"momos2d_wa_r{rows}_c{cols}_cap{cap}.pdf")


# %%
runs = [
    "model-jrgwsa5u",
]
_config_cache = {}
for run_name in runs:
    report = Report()
    rows = cols = cap = None
    for i in trange(0, 1):
        artifact = api.artifact(
            f"danesinoo-university-of-copenhagen/momos-reproduction/{run_name}:v{i}"
        )
        if run_name not in _config_cache:
            q_cfg = artifact.logged_by().config.get("quantization", {})
            _config_cache[run_name] = q_cfg
        q_cfg = _config_cache[run_name]
        rows, cols, cap = q_cfg["rows"], q_cfg["cols"], q_cfg["capacity"]
        d = artifact.download()
        ckpt_path = os.path.join(d, "model.ckpt")
        if os.path.exists(ckpt_path):
            report.append_figures(plot_weights_2d(ckpt_path, rows, cols, 100))
    if rows is not None:
        report.save(f"momos2d_wa_r{rows}_c{cols}_cap{cap}.pdf")

# %%

import wandb

api = wandb.Api()
artifact = api.artifact(
    "danesinoo-university-of-copenhagen/momos2d-remake/model-sh4p5gxq:v0"
)
artifact_dir = artifact.download()
