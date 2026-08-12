# %% [markdown]
# %%
import wandb
from src.view import Report
from src.view.weight_distribution import plot_weights_2d
import os
from tqdm import trange

project = "momos2d-remake"  # "momos-reproduction"
api = wandb.Api()
runs = [
    "model-pw0ti3vz",
    # "model-qcci4krm",
    # "model-y4jvu3xv"
]
_config_cache = {}
for run_name in runs:
    report = Report()
    rows = cols = cap = None
    for i in trange(0, 20):
        # Deterministic local path so we can check for an existing download *before*
        # touching the network, instead of relying on wandb's default cache directory.
        root = os.path.join("artifacts", f"{run_name}-v{i}")
        ckpt_path = os.path.join(root, "model.ckpt")
        need_artifact = run_name not in _config_cache or not os.path.exists(ckpt_path)

        if need_artifact:
            artifact = api.artifact(
                f"danesinoo-university-of-copenhagen/momos-collapse/{run_name}:v{i}"
            )
            if run_name not in _config_cache:
                q_cfg = artifact.logged_by().config.get("quantization", {})
                _config_cache[run_name] = q_cfg
            if not os.path.exists(ckpt_path):
                artifact.download(root=root)

        q_cfg = _config_cache[run_name]
        rows, cols, cap = q_cfg["rows"], q_cfg["cols"], q_cfg["capacity"]
        if os.path.exists(ckpt_path):
            report.append_figures(plot_weights_2d(ckpt_path, rows, cols, i * 20))
        for figure in report.figures:
            figure.fontsize = 13

    if rows is not None:
        report.save(f"momos2d_wa_r{rows}_c{cols}_cap{cap}.pdf")
