"""Smoke test for MotifMaskDataModule + LitMamba end-to-end.

Requires a momos2d-quantized MLP checkpoint at:
    artifacts/model-sh4p5gxq:v0/model.ckpt
Skipped if the checkpoint is missing (e.g. on CI).
"""

import os
import pytest
import torch

from data.momos_mask_datamodule import MotifMaskDataModule
from model.lit_mamba import LitMamba


CHECKPOINT = "artifacts/model-sh4p5gxq:v0/model.ckpt"

_MAMBA_TEST_KWARGS = {
    "hidden_size": 16,
    "num_heads": 2,
    "head_dim": 16,
    "state_size": 8,
    "num_hidden_layers": 1,
    "n_groups": 1,
    "hidden_act": "silu",
    "time_step_min": 0.001,
    "time_step_max": 0.1,
    "cnn_kernel_size": 1,
    "cnn_padding": 0,
}


@pytest.fixture(scope="module")
def datamodule():
    if not os.path.exists(CHECKPOINT):
        pytest.skip(f"Checkpoint not found: {CHECKPOINT}")
    dm = MotifMaskDataModule(
        checkpoint_path=CHECKPOINT,
        rows=1,
        cols=2,
        motif_rows=1,
        motif_cols=2,
    )
    dm.setup()
    return dm


def test_summary_has_data_fields(datamodule):
    expected = {
        "n_motifs",
        "num_rows",
        "num_cols",
        "num_layers",
        "cnn_out_channels",
    }
    assert set(datamodule.summary().keys()) == expected


def test_dataset_sample_shape(datamodule):
    ds = datamodule.train_dataset
    (motif, layer, n_rows, n_cols), target = ds[0]
    assert motif.dtype == torch.long and motif.ndim == 0
    assert layer.dtype == torch.long and layer.ndim == 0
    assert target.shape == (
        int(n_rows) * int(n_cols),
        datamodule.rows * datamodule.cols,
    )
    assert target.dtype == torch.float32
    assert ((target == 0) | (target == 1)).all()


def test_forward_and_training_step_run(datamodule):
    model = LitMamba(**datamodule.summary(), **_MAMBA_TEST_KWARGS, epochs=1)
    model.log = lambda *args, **kwargs: None  # type: ignore
    batch = next(iter(datamodule.train_dataloader()))
    loss = model.training_step(batch, 0)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
