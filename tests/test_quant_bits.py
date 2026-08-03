"""Unit tests for theoretical bit calculation and compression rate utilities."""

import pytest
import torch
import torch.nn as nn
from utils.quant_bits import count_trainable_parameters, compute_quantization_bits


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 16, bias=False)  # 256 params
        self.fc2 = nn.Linear(16, 16, bias=False)  # 256 params
        # total params = 512


def test_count_trainable_parameters():
    model = DummyModel()
    assert count_trainable_parameters(model) == 512


def test_compute_quantization_bits_none():
    model = DummyModel()
    res = compute_quantization_bits(model, {"enabled": False})
    assert res["num_parameters"] == 512
    assert res["dense_bits"] == 512 * 32
    assert res["compressed_bits"] == 512 * 32
    assert res["compression_rate"] == 1.0
    assert res["bpp"] == 32.0


def test_compute_quantization_bits_qat():
    model = DummyModel()
    res = compute_quantization_bits(model, {"enabled": True, "method": "qat", "q": 4})
    assert res["compressed_bits"] == 512 * 4
    assert res["compression_rate"] == 8.0
    assert res["bpp"] == 4.0


def test_compute_quantization_bits_momos2d():
    model = DummyModel()
    # 512 params. block_size = 4x4 = 16. Total blocks = 512 / 16 = 32 blocks.
    # capacity = 0.25 => k = 8 motifs.
    # motif bits = 8 * 16 * 32 = 4096 bits.
    # index bits = 32 blocks * ceil(log2(8)) = 32 * 3 = 96 bits.
    # compressed_bits = 4096 + 96 = 4192 bits.
    res = compute_quantization_bits(
        model,
        {
            "enabled": True,
            "method": "momos2d",
            "rows": 4,
            "cols": 4,
            "capacity": 0.25,
            "q_bits": 32,
        },
    )
    assert res["motif_bits"] == 4096.0
    assert res["index_bits"] == 96.0
    assert res["compressed_bits"] == 4192.0
    assert res["compression_rate"] == (512 * 32) / 4192.0


def test_compute_quantization_bits_hierarchical():
    model = DummyModel()
    # Primary: 4x4 = 16 size, 32 blocks. capacity 0.5 => k1 = 16 motifs.
    # Secondary: 2x2 = 4 primary motifs per big block. n_big = ceil(32 / 4) = 8.
    # Secondary capacity 0.5 => k_per_bucket = round(16 * 0.5) = 8.
    res = compute_quantization_bits(
        model,
        {
            "enabled": True,
            "method": "hierarchical_momos2d",
            "primary": {"rows": 4, "cols": 4, "capacity": 0.5},
            "secondary": {"rows": 2, "cols": 2, "capacity": 0.5},
            "q_bits": 32,
        },
    )
    assert res["compressed_bits"] > 0
    assert res["compression_rate"] > 1.0
