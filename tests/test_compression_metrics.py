"""Tests for compression metrics computation."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.view.compression_metrics import (
    _compute_rac,
    compute_compression_metrics,
)


def test_compute_rac_momos():
    """Test RAC computation for MoMos model."""
    run = {
        "config": {
            "s": 2,
            "k": 2,
            "q": 4,
            "n": 256,
            "m": 2,
        },
        "metrics": {},
    }

    rac, rac_qat = _compute_rac(run)

    # RAC = n*q / (k*s*q + m*ceil(log2(k)))
    #    = 256*4 / (2*2*4 + 2*ceil(log2(2)))
    #    = 1024 / (16 + 2*1) = 1024 / 18 = 56.89
    expected_rac = 256 * 4 / (2 * 2 * 4 + 2 * round(1.0))
    assert abs(rac - expected_rac) < 0.01, f"Expected {expected_rac}, got {rac}"

    # QAT: rac_qat = 32/q = 32/4 = 8
    assert rac_qat == 8.0, f"Expected 8.0, got {rac_qat}"

    print("✓ test_compute_rac_momos passed")


def test_compute_rac_fp32():
    """Test RAC for FP32 model (default q=16)."""
    run = {
        "config": {
            "s": 1,
            "k": 1,
            "q": 16,
            "n": 256,
            "m": 2,
        },
        "metrics": {},
    }

    rac, rac_qat = _compute_rac(run)

    # RAC = 256*16 / (1*1*16 + 2*ceil(log2(1)))
    #    = 4096 / (16 + 0) = 256
    expected_rac = 256 * 16 / (1 * 1 * 16 + 2 * round(0))
    assert abs(rac - expected_rac) < 0.01, f"Expected {expected_rac}, got {rac}"

    # QAT: rac_qat = None for q=16 (FP32 baseline)
    assert rac_qat is None, f"Expected None, got {rac_qat}"

    print("✓ test_compute_rac_fp32 passed")


def test_compute_compression_metrics_with_metrics():
    """Test compute_compression_metrics with BDM metrics."""
    run = {
        "config": {
            "s": 2,
            "k": 2,
            "q": 4,
            "n": 256,
            "m": 2,
        },
        "metrics": {
            "metrics/bdm_complexity": 0.85,
            "metrics/bdm_complexity_ratio": 1.2,
        },
    }

    result = compute_compression_metrics(run)

    assert "rac" in result
    assert "rac_qat" in result
    assert "bdm_complexity" in result
    assert "bdm_ratio" in result

    assert result["rac"] == 56.888888888888886
    assert result["rac_qat"] == 8.0
    assert result["bdm_complexity"] == 0.85
    assert result["bdm_ratio"] == 1.2

    print("✓ test_compute_compression_metrics_with_metrics passed")


def test_compute_compression_metrics_without_bdm():
    """Test compute_compression_metrics without BDM metrics."""
    run = {
        "config": {
            "s": 2,
            "k": 2,
            "q": 4,
            "n": 256,
            "m": 2,
        },
        "metrics": {},
    }

    result = compute_compression_metrics(run)

    assert result["rac"] == 56.888888888888886
    assert result["rac_qat"] == 8.0
    assert result["bdm_complexity"] is None
    assert result["bdm_ratio"] is None

    print("✓ test_compute_compression_metrics_without_bdm passed")


def test_compute_rac_edge_cases():
    """Test RAC computation with edge cases."""
    # Test with k=1
    run1 = {"config": {"s": 1, "k": 1, "q": 8, "n": 128, "m": 1}, "metrics": {}}
    rac1, _ = _compute_rac(run1)
    # RAC = 128*8 / (1*1*8 + 1*ceil(log2(1))) = 1024 / 8 = 128
    expected1 = 128 * 8 / (1 * 1 * 8 + 1 * round(0))
    assert abs(rac1 - expected1) < 0.01

    # Test with larger k
    run2 = {"config": {"s": 2, "k": 4, "q": 8, "n": 512, "m": 2}, "metrics": {}}
    rac2, _ = _compute_rac(run2)
    # RAC = 512*8 / (4*2*8 + 2*ceil(log2(4))) = 4096 / (64 + 2*2) = 4096 / 68 = 60.24
    expected2 = 512 * 8 / (4 * 2 * 8 + 2 * round(2.0))
    assert abs(rac2 - expected2) < 0.01

    print("✓ test_compute_rac_edge_cases passed")


if __name__ == "__main__":
    test_compute_rac_momos()
    test_compute_rac_fp32()
    test_compute_compression_metrics_with_metrics()
    test_compute_compression_metrics_without_bdm()
    test_compute_rac_edge_cases()
    print("\n✅ All tests passed!")
