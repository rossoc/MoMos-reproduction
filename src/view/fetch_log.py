import wandb
from wandb.apis.public import Runs

from dotenv import load_dotenv
from os import getenv

import pandas as pd

load_dotenv()


QUANT_METRICS = [
    "metrics/bdm_complexity",
    "metrics/gzip_compression_rate",
    "metrics/bz2_compression_rate",
    "metrics/lzma_compression_rate",
    "metrics/sparsity",
    "metrics/weight_l2",
]

MOMOS_METRICS = [
    "metrics/bdm_complexity",
    "metrics/gzip_compression_rate",
    "metrics/bz2_compression_rate",
    "metrics/lzma_compression_rate",
    "metrics/weight_l2",
    "metrics/sparsity",
    "quant/distortion",
    "quant/changed_weights",
]


ORIGINAL_FIELDS_MAPPING = {
    "val_loss": "val/loss",
    "val_acc": "val/acc",
    "train_acc": "train/acc",
    "train_loss": "train/loss",
}


def fetch_runs(
    entity=None,
    project=None,
    rename_dict=ORIGINAL_FIELDS_MAPPING,
    quant_metrics=QUANT_METRICS,
    momos_metrics=MOMOS_METRICS,
    skip_n=0,
    pop_n=0,
):
    api = wandb.Api()
    entity = entity or getenv("WANDB_ENTITY", "")
    project = project or getenv("WANDB_PROJECT", "")

    runs = api.runs(entity + "/" + project, lazy=False)
    return _runs_clean_up(
        runs, rename_dict, quant_metrics, momos_metrics, skip_n, pop_n
    )


def _runs_clean_up(
    runs: Runs, rename_dict, quant_metrics, momos_metrics, skip_n=0, pop_n=0
):
    results = []
    quant_models = 0
    standard_models = 0
    momos_models = 0

    for run in runs[skip_n : len(runs) - pop_n]:
        history = run.history(samples=500)
        assert type(history) is pd.DataFrame

        try:
            history = history.rename(columns=rename_dict)

            metrics = []
            metrics += [history[["epoch"]].dropna().drop_duplicates()]
            metrics += extract_columns(history, rename_dict.values())
            standard_models += 1

            try:
                test_acc = history["test/acc"].dropna().iloc[0]
            except Exception as _:
                test_acc = 0

            try:
                metrics += extract_columns(history, quant_metrics)
                quant_models += 1
            except Exception as _:
                pass

            try:
                metrics += extract_columns(history, momos_metrics)
                momos_models += 1
            except Exception as _:
                pass

        except Exception as e:
            print(history.columns)
            print(e)
            break

        metrics = merge_dfs(metrics)
        results.append(
            {
                "name": run.name,
                "metrics": metrics,
                "config": run.config,
                "test_acc": test_acc,
            }
        )

    standard_models -= quant_models
    quant_models -= momos_models
    return results, standard_models, quant_models, momos_models


def extract_columns(df, columns, key="epoch"):
    return [df[["epoch", column]].dropna(how="any") for column in columns]


def merge_dfs(dfs):
    res = dfs[0]

    for df in dfs[1:]:
        res = pd.merge(res, df, how="outer")
    return res
