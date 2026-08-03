# %% [markdown]
# %%
import pandas as pd
import wandb
from src.view import Report, extract_columns, merge_dfs
from src.view.weight_distribution import plot_weights_2d
import ast
import numpy as np
import os
from tqdm import tqdm, trange
from collections import defaultdict

project = "momos2d-remake"  # "momos-reproduction"
api = wandb.Api()
runs = [
    # "model-pw0ti3vz",
    # "model-qcci4krm",
    "model-y4jvu3xv"
]
_config_cache = {}
for run_name in runs:
    report = Report()
    rows = cols = cap = None
    for i in trange(0, 20):
        artifact = api.artifact(
            f"danesinoo-university-of-copenhagen/momos-collapse/{run_name}:v{i}"
        )
        if run_name not in _config_cache:
            q_cfg = artifact.logged_by().config.get("quantization", {})  # type: ignore
            _config_cache[run_name] = q_cfg
        q_cfg = _config_cache[run_name]
        rows, cols, cap = q_cfg["rows"], q_cfg["cols"], q_cfg["capacity"]
        d = artifact.download()
        ckpt_path = os.path.join(d, "model.ckpt")
        if os.path.exists(ckpt_path):
            run_data = (ckpt_path, rows, cols, cap, i * 20)
            report.append_figures(plot_weights_2d(ckpt_path, rows, cols, cap, i * 20))

            if cols != rows:
                report.append_figures(
                    plot_weights_2d(ckpt_path, cols, rows, cap, i * 20)
                )
    if rows is not None:
        report.save(f"momos2d_wa_r{rows}_c{cols}_cap{cap}.pdf")
