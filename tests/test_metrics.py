"""Tests for metrics module - ensures refactored code produces identical results."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.utils.metrics import (
    flatten_weights,
    compute_metrics,
    get_compression_payload_from_weights,
    compression_rate,
)

from .old_metrics import (
    compute_sparsity,
    compute_l2,
    compute_gzip,
    compute_bz2,
    compute_lzma,
    compute_bdm,
    registry,
)


# --- Helper: create simple models for testing ---


class SimpleModel(nn.Module):
    """Model with known weights for deterministic testing."""

    def __init__(self, weights_list):
        super().__init__()
        self.layers = nn.ParameterList(
            [nn.Parameter(torch.tensor(w, dtype=torch.float32)) for w in weights_list]
        )

    def forward(self, x):
        return x


class EmptyModel(nn.Module):
    """Model with no trainable parameters."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class ZeroWeightModel(nn.Module):
    """Model with all zero weights."""

    def __init__(self, size=10):
        super().__init__()
        self.linear = nn.Linear(size, size, bias=False)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def model_with_known_weights():
    """Returns a model with known weight values for deterministic testing."""
    # Create a simple model with known weights
    model = nn.Sequential(
        nn.Linear(4, 3, bias=False),
        nn.Linear(3, 2, bias=False),
    )
    # Set known weights
    with torch.no_grad():
        model[0].weight.data = torch.tensor(
            [
                [1.0, 0.0, -1.0, 0.5],
                [0.0, 0.0, 0.0, 0.0],
                [0.5, -0.5, 0.5, -0.5],
            ]
        )
        model[1].weight.data = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
            ]
        )
    return model


@pytest.fixture
def empty_model():
    return EmptyModel()


@pytest.fixture
def zero_weight_model():
    return ZeroWeightModel(size=5)


# --- Tests for flatten_weights ---


class TestFlattenWeights:
    def test_flatten_known_weights(self, model_with_known_weights):
        weights = flatten_weights(model_with_known_weights)
        assert isinstance(weights, np.ndarray)
        assert weights.dtype == np.float32
        # 4*3 + 3*2 = 12 + 6 = 18 parameters
        assert len(weights) == 18

    def test_flatten_empty_model(self, empty_model):
        weights = flatten_weights(empty_model)
        assert len(weights) == 0
        assert weights.dtype == np.float32

    def test_flatten_preserves_values(self):
        model = SimpleModel(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[0.5, -0.5]],
            ]
        )
        weights = flatten_weights(model)
        expected = np.array([1.0, 2.0, 3.0, 4.0, 0.5, -0.5], dtype=np.float32)
        np.testing.assert_array_equal(weights, expected)


# --- Tests for individual metric functions ---


class TestComputeSparsity:
    def test_sparsity_with_zeros(self, zero_weight_model):
        result = compute_sparsity(zero_weight_model)
        assert result == {"sparsity": 1.0}

    def test_sparsity_no_zeros(self):
        model = SimpleModel([[[1.0, 2.0, 3.0]]])
        result = compute_sparsity(model)
        assert result["sparsity"] == 0.0

    def test_sparsity_mixed(self, model_with_known_weights):
        result = compute_sparsity(model_with_known_weights)
        # Count zeros in the known weights
        weights = flatten_weights(model_with_known_weights)
        expected_sparsity = float((weights == 0).mean())
        assert result["sparsity"] == pytest.approx(expected_sparsity, abs=1e-7)

    def test_sparsity_empty_model(self, empty_model):
        result = compute_sparsity(empty_model)
        assert result == {"sparsity": 0.0}


class TestComputeL2:
    def test_l2_known_weights(self, model_with_known_weights):
        result = compute_l2(model_with_known_weights)
        weights = flatten_weights(model_with_known_weights)
        expected_l2 = float(np.linalg.norm(weights))
        assert result["weight_l2"] == pytest.approx(expected_l2, abs=1e-6)

    def test_l2_empty_model(self, empty_model):
        result = compute_l2(empty_model)
        assert result == {"weight_l2": 0.0}

    def test_l2_zero_model(self, zero_weight_model):
        result = compute_l2(zero_weight_model)
        assert result["weight_l2"] == pytest.approx(0.0, abs=1e-7)


class TestCompressionMetrics:
    @pytest.mark.parametrize("binarized", [False, True])
    def test_gzip_returns_valid_rate(self, model_with_known_weights, binarized):
        result = compute_gzip(model_with_known_weights, compression_binarized=binarized)
        assert "gzip_compression_rate" in result
        assert result["gzip_compression_rate"] > 0

    @pytest.mark.parametrize("binarized", [False, True])
    def test_bz2_returns_valid_rate(self, model_with_known_weights, binarized):
        result = compute_bz2(model_with_known_weights, compression_binarized=binarized)
        assert "bz2_compression_rate" in result
        assert result["bz2_compression_rate"] > 0

    @pytest.mark.parametrize("binarized", [False, True])
    def test_lzma_returns_valid_rate(self, model_with_known_weights, binarized):
        result = compute_lzma(model_with_known_weights, compression_binarized=binarized)
        assert "lzma_compression_rate" in result
        assert result["lzma_compression_rate"] > 0

    def test_compression_empty_model(self, empty_model):
        # Should handle empty model gracefully
        result_gzip = compute_gzip(empty_model)
        result_bz2 = compute_bz2(empty_model)
        result_lzma = compute_lzma(empty_model)
        # Empty payload returns 0.0 compression rate
        assert result_gzip["gzip_compression_rate"] == 0.0
        assert result_bz2["bz2_compression_rate"] == 0.0
        assert result_lzma["lzma_compression_rate"] == 0.0


class TestBDM:
    def test_bdm_returns_value(self, model_with_known_weights):
        result = compute_bdm(model_with_known_weights)
        assert "bdm_complexity" in result
        # BDM should either return a float or None (if engine unavailable)
        if result["bdm_complexity"] is not None:
            assert isinstance(result["bdm_complexity"], float)
            assert result["bdm_complexity"] >= 0

    def test_bdm_empty_model(self, empty_model):
        result = compute_bdm(empty_model)
        assert result == {"bdm_complexity": 0.0}


# --- Tests for compression payload and rate ---


class TestCompressionPayload:
    def test_payload_non_empty(self, model_with_known_weights):
        weights = flatten_weights(model_with_known_weights)
        payload = get_compression_payload_from_weights(
            weights, compression_binarized=False
        )
        assert isinstance(payload, bytes)
        assert len(payload) > 0

    def test_payload_binarized(self):
        weights = np.array([1.0, -1.0, 0.0, 0.5], dtype=np.float32)
        payload = get_compression_payload_from_weights(
            weights, compression_binarized=True
        )
        # Binarized: (weights > 0) -> [1, 0, 0, 1]
        assert len(payload) == len(weights)  # 1 byte per uint8

    def test_payload_empty_weights(self):
        weights = np.array([], dtype=np.float32)
        payload = get_compression_payload_from_weights(
            weights, compression_binarized=False
        )
        assert payload == b""

    def test_compression_rate_empty(self):
        assert compression_rate(b"", b"") == 0.0

    def test_compression_rate_basic(self):
        payload = b"test data" * 100
        import gzip

        compressed = gzip.compress(payload)
        rate = compression_rate(payload, compressed)
        assert rate > 0


# --- Tests for compute_metrics batch function ---


class TestComputeMetrics:
    def test_compute_all_metrics(self, model_with_known_weights):
        names = list(registry.keys())
        result = compute_metrics(model_with_known_weights, names)
        # Should have all metric keys
        assert "sparsity" in result
        assert "weight_l2" in result
        assert "bdm_complexity" in result
        assert "gzip_compression_rate" in result
        assert "bz2_compression_rate" in result
        assert "lzma_compression_rate" in result

    def test_compute_metrics_subset(self, model_with_known_weights):
        result = compute_metrics(model_with_known_weights, ["sparsity", "l2"])
        assert set(result.keys()) == {"sparsity", "weight_l2"}

    def test_compute_metrics_unknown_raises(self, model_with_known_weights):
        with pytest.raises(ValueError, match="Unknown metric"):
            compute_metrics(model_with_known_weights, ["invalid_metric"])

    def test_compute_metrics_empty_model(self, empty_model):
        result = compute_metrics(empty_model, ["sparsity", "l2"])
        assert result["sparsity"] == 0.0
        assert result["weight_l2"] == 0.0


# --- CRITICAL: Batch vs Individual consistency tests ---


class TestBatchVsIndividualConsistency:
    """These tests ensure compute_metrics() returns the EXACT same values
    as calling individual compute_*() functions."""

    def test_sparsity_consistency(self, model_with_known_weights):
        model = model_with_known_weights
        # Individual
        individual = compute_sparsity(model)
        # Batch
        batch = compute_metrics(model, ["sparsity"])
        assert batch["sparsity"] == pytest.approx(individual["sparsity"], abs=1e-7)

    def test_l2_consistency(self, model_with_known_weights):
        model = model_with_known_weights
        individual = compute_l2(model)
        batch = compute_metrics(model, ["l2"])
        assert batch["weight_l2"] == pytest.approx(individual["weight_l2"], abs=1e-6)

    def test_gzip_consistency(self, model_with_known_weights):
        model = model_with_known_weights
        for binarized in [False, True]:
            individual = compute_gzip(model, compression_binarized=binarized)
            batch = compute_metrics(model, ["gzip"], compression_binarized=binarized)
            assert batch["gzip_compression_rate"] == pytest.approx(
                individual["gzip_compression_rate"], abs=1e-7
            )

    def test_bz2_consistency(self, model_with_known_weights):
        model = model_with_known_weights
        for binarized in [False, True]:
            individual = compute_bz2(model, compression_binarized=binarized)
            batch = compute_metrics(model, ["bz2"], compression_binarized=binarized)
            assert batch["bz2_compression_rate"] == pytest.approx(
                individual["bz2_compression_rate"], abs=1e-7
            )

    def test_lzma_consistency(self, model_with_known_weights):
        model = model_with_known_weights
        for binarized in [False, True]:
            individual = compute_lzma(model, compression_binarized=binarized)
            batch = compute_metrics(model, ["lzma"], compression_binarized=binarized)
            assert batch["lzma_compression_rate"] == pytest.approx(
                individual["lzma_compression_rate"], abs=1e-7
            )

    def test_bdm_consistency(self, model_with_known_weights):
        model = model_with_known_weights
        individual = compute_bdm(model)
        batch = compute_metrics(model, ["bdm"])
        if individual["bdm_complexity"] is None:
            assert batch["bdm_complexity"] is None
        else:
            assert batch["bdm_complexity"] == pytest.approx(
                individual["bdm_complexity"], abs=1e-6
            )

    def test_all_metrics_consistency(self, model_with_known_weights):
        """Compute all metrics both ways and verify they match."""
        model = model_with_known_weights
        names = ["sparsity", "l2", "gzip", "bz2", "lzma", "bdm"]

        # Batch result
        batch_result = compute_metrics(model, names)

        # Individual results
        individual_results = {
            "sparsity": compute_sparsity(model),
            "l2": compute_l2(model),
            "gzip": compute_gzip(model),
            "bz2": compute_bz2(model),
            "lzma": compute_lzma(model),
            "bdm": compute_bdm(model),
        }

        # Merge individual results
        merged = {}
        for result in individual_results.values():
            merged.update(result)

        # Compare
        for key in merged:
            if merged[key] is None:
                assert batch_result[key] is None, f"Mismatch for {key}"
            else:
                assert batch_result[key] == pytest.approx(merged[key], abs=1e-6), (
                    f"Mismatch for {key}: batch={batch_result[key]}, individual={merged[key]}"
                )

    @pytest.mark.parametrize("binarized", [False, True])
    def test_compression_metrics_with_binarization(
        self, model_with_known_weights, binarized
    ):
        """Ensure compression metrics are consistent with binarization flag."""
        model = model_with_known_weights
        names = ["gzip", "bz2", "lzma"]

        batch = compute_metrics(model, names, compression_binarized=binarized)

        for name in names:
            fn = registry[name]
            individual = fn(model, compression_binarized=binarized)
            key = list(individual.keys())[0]
            assert batch[key] == pytest.approx(individual[key], abs=1e-7)


# --- Edge case tests ---


class TestEdgeCases:
    def test_single_parameter_model(self):
        model = SimpleModel([[[1.0, 2.0, 3.0]]])
        result = compute_metrics(model, ["sparsity", "l2"])
        assert result["sparsity"] == 0.0
        assert result["weight_l2"] > 0

    def test_mixed_zero_weights_model(self):
        model = nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            model.weight.data = torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 2.0, 3.0],
                    [0.0, 0.0, 0.0],
                ]
            )
        result = compute_sparsity(model)
        # 6 out of 9 are zero
        assert result["sparsity"] == pytest.approx(6.0 / 9.0, abs=1e-7)

    def test_multiple_compute_metrics_calls_same_result(self, model_with_known_weights):
        """Ensure calling compute_metrics multiple times gives same result."""
        model = model_with_known_weights
        names = ["sparsity", "l2", "gzip"]
        result1 = compute_metrics(model, names)
        result2 = compute_metrics(model, names)
        assert result1 == result2

    def test_registry_contains_all_expected_metrics(self):
        expected = {"sparsity", "l2", "bdm", "gzip", "bz2", "lzma"}
        assert set(registry.keys()) == expected

    def test_binarized_vs_nonbinarized_payload_differ(self, model_with_known_weights):
        """Ensure binarized and non-binarized payloads are different."""
        weights = flatten_weights(model_with_known_weights)
        payload_normal = get_compression_payload_from_weights(
            weights, compression_binarized=False
        )
        payload_binarized = get_compression_payload_from_weights(
            weights, compression_binarized=True
        )
        assert payload_normal != payload_binarized
        # Binarized should be smaller (1 byte per weight vs 4 bytes per float32)
        assert len(payload_binarized) < len(payload_normal)
