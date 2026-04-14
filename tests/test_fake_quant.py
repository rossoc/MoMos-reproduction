"""Tests for fake quantization components."""

import pytest
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

from src.quantizers.fake_quant import (
    RoundSTE,
    UniformSymmetric,
    FakeQuantParametrization,
    _get_fake_quant_parametrization,
    attach_weight_quantizers,
    toggle_quantization,
    quantize_qat,
)


class SimpleModel(nn.Module):
    """Model with two linear layers for testing."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        return self.fc2(self.fc1(x))


class TestRoundSTE:
    def test_forward_rounds_values(self):
        x = torch.tensor([1.2, 1.7, -0.5, 2.9])
        result = RoundSTE.apply(x)
        expected = torch.tensor([1.0, 2.0, -0.0, 3.0])
        torch.testing.assert_close(result, expected)

    def test_backward_pass_through(self):
        x = torch.tensor([1.5, 2.3], requires_grad=True)
        y = RoundSTE.apply(x)
        y.sum().backward()
        # Gradient should pass through unchanged
        assert x.grad is not None
        torch.testing.assert_close(x.grad, torch.ones_like(x))


class TestUniformSymmetric:
    def test_identity_32bit(self):
        weight = torch.randn(4, 4)
        quantizer = UniformSymmetric(bitwidth=32)
        result = quantizer(weight)
        torch.testing.assert_close(result, weight)

    def test_quantize_8bit(self):
        weight = torch.tensor([[1.0, -2.0], [0.5, -1.5]])
        quantizer = UniformSymmetric(bitwidth=8)
        result = quantizer(weight)
        # Values should be quantized to 256 levels
        assert result.shape == weight.shape
        # absmax=2.0, scale=2.0/127, so values are snapped to discrete levels
        assert not torch.allclose(result, weight)

    def test_zero_weight_unchanged(self):
        weight = torch.zeros(4, 4)
        quantizer = UniformSymmetric(bitwidth=8)
        result = quantizer(weight)
        torch.testing.assert_close(result, weight)

    def test_invalid_bitwidth(self):
        quantizer = UniformSymmetric(bitwidth=1)
        weight = torch.randn(4, 4)
        with pytest.raises(ValueError, match="bitwidth must be >= 2"):
            quantizer(weight)


class TestHasFakeQuantWeightParametrization:
    def test_no_parametrizations(self):
        linear = nn.Linear(4, 4)
        assert not _get_fake_quant_parametrization(linear)

    def test_has_fake_quant(self):
        linear = nn.Linear(4, 4)
        fq = FakeQuantParametrization(UniformSymmetric(bitwidth=8))
        parametrize.register_parametrization(linear, "weight", fq)
        assert bool(_get_fake_quant_parametrization(linear))

    def test_has_non_fake_quant(self):
        linear = nn.Linear(4, 4)

        # Simple parametrization that just returns weight as-is
        class IdentityParam(nn.Module):
            def forward(self, w):
                return w

        parametrize.register_parametrization(linear, "weight", IdentityParam())
        assert not _get_fake_quant_parametrization(linear)


class TestAttachWeightQuantizers:
    def test_attaches_to_linear_layers(self):
        model = SimpleModel()
        result = attach_weight_quantizers(model, bitwidth=8)
        assert result["attached_modules"] == 2  # fc1 and fc2
        assert result["updated_modules"] == 0

    def test_attaches_to_all_trainable_weights(self):
        class ModelWithConv(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 8, 3)
                self.fc = nn.Linear(8, 2)

            def forward(self, x):
                return self.fc(x)

        model = ModelWithConv()
        result = attach_weight_quantizers(model, bitwidth=8)
        assert result["attached_modules"] == 2

    def test_excludes_layers_by_name(self):
        model = SimpleModel()
        result = attach_weight_quantizers(model, bitwidth=8, exclude_layers=["fc1"])
        assert result["attached_modules"] == 1  # only fc2
        assert bool(_get_fake_quant_parametrization(model.fc2))
        assert not _get_fake_quant_parametrization(model.fc1)

    def test_case_insensitive_exclusion(self):
        model = SimpleModel()
        result = attach_weight_quantizers(
            model, bitwidth=8, exclude_layers=["FC1", "FC2"]
        )
        assert result["attached_modules"] == 0

    def test_updates_existing_quantizers(self):
        model = SimpleModel()
        # First attachment
        result1 = attach_weight_quantizers(model, bitwidth=8)
        assert result1["attached_modules"] == 2

        # Second call should update, not attach
        result2 = attach_weight_quantizers(model, bitwidth=4)
        assert result2["attached_modules"] == 0
        assert result2["updated_modules"] == 2

        # Verify bitwidth was updated
        for p in model.fc1.parametrizations.weight:
            if isinstance(p, FakeQuantParametrization):
                assert p.quantizer.bitwidth == 4

    def test_enabled_flag(self):
        model = SimpleModel()
        attach_weight_quantizers(model, bitwidth=8, enabled=False)
        for p in model.fc1.parametrizations.weight:
            if isinstance(p, FakeQuantParametrization):
                assert not p.enabled

    def test_skips_non_trainable_weights(self):
        model = SimpleModel()
        model.fc1.weight.requires_grad_(False)
        result = attach_weight_quantizers(model, bitwidth=8)
        assert result["attached_modules"] == 1  # only fc2

    def test_skips_empty_weights(self):
        model = nn.Module()
        model.weight = nn.Parameter(torch.empty(0))
        result = attach_weight_quantizers(model, bitwidth=8)
        assert result["attached_modules"] == 0

    def test_does_not_override_non_fake_parametrizations(self):
        linear = nn.Linear(4, 4)

        class IdentityParam(nn.Module):
            def forward(self, w):
                return w

        parametrize.register_parametrization(linear, "weight", IdentityParam())

        model = nn.Sequential(linear)
        result = attach_weight_quantizers(model, bitwidth=8)
        # Should not attach because weight is already parametrized
        assert result["attached_modules"] == 0


class TestToggleQuantization:
    def test_enable_quantization(self):
        model = SimpleModel()
        attach_weight_quantizers(model, bitwidth=8, enabled=False)
        toggled = toggle_quantization(model, enabled=True)
        assert toggled == 2
        for p in model.fc1.parametrizations.weight:
            if isinstance(p, FakeQuantParametrization):
                assert p.enabled

    def test_disable_quantization(self):
        model = SimpleModel()
        attach_weight_quantizers(model, bitwidth=8, enabled=True)
        toggled = toggle_quantization(model, enabled=False)
        assert toggled == 2
        for p in model.fc1.parametrizations.weight:
            if isinstance(p, FakeQuantParametrization):
                assert not p.enabled

    def test_no_quantizers(self):
        model = SimpleModel()
        toggled = toggle_quantization(model, enabled=True)
        assert toggled == 0


class TestPrepareQAT:
    def test_enables_qat(self):
        model = SimpleModel()
        # Attach quantizers first
        attach_weight_quantizers(model, bitwidth=8)
        result = quantize_qat(model, {"q": 8})
        assert result["qat_enabled"] is True
        assert result["q_bits"] == 8

    def test_disables_32bit(self):
        model = SimpleModel()
        attach_weight_quantizers(model, bitwidth=8)
        result = quantize_qat(model, {"q": 32})
        assert result["qat_enabled"] is False
        assert result["disabled_modules"] > 0

    def test_invalid_bitwidth(self):
        model = SimpleModel()
        with pytest.raises(ValueError, match="q must be >= 2"):
            quantize_qat(model, {"q": 1})

    def test_exclude_layers(self):
        model = SimpleModel()
        attach_weight_quantizers(model, bitwidth=8)
        result = quantize_qat(model, {"q": 4, "exclude_layers": ["fc1"]})
        assert result["qat_enabled"] is True
        # fc1 should have bitwidth 8 (original), fc2 should have 4
        fc1_bits = None
        fc2_bits = None
        for p in model.fc1.parametrizations.weight:
            if isinstance(p, FakeQuantParametrization):
                fc1_bits = p.quantizer.bitwidth
        for p in model.fc2.parametrizations.weight:
            if isinstance(p, FakeQuantParametrization):
                fc2_bits = p.quantizer.bitwidth
        assert fc1_bits == 8  # excluded, not updated
        assert fc2_bits == 4  # updated


class TestQuantizeQAT:
    def test_calls_prepare_qat(self):
        model = SimpleModel()
        attach_weight_quantizers(model, bitwidth=8)
        result = quantize_qat(model, {"q": 8})
        assert result["qat_enabled"] is True
