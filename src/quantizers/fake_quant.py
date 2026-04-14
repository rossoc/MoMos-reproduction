"""Fake quantization components for QAT (Quantization-Aware Training)."""

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize


class RoundSTE(torch.autograd.Function):
    """Straight-through rounding estimator used by fake quantization."""

    @staticmethod
    def forward(ctx, input_tensor):
        return torch.round(input_tensor)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        return grad_output


class UniformSymmetric(nn.Module):
    """Per-tensor signed symmetric fake quantizer."""

    def __init__(self, bitwidth=8):
        super().__init__()
        self.bitwidth = int(bitwidth)

    def forward(self, weight):
        bits = int(self.bitwidth)
        if bits >= 32:
            return weight
        if bits < 2:
            raise ValueError(f"bitwidth must be >= 2, got {bits}")
        q = 2 ** (bits - 1) - 1
        with torch.no_grad():
            absmax = torch.max(torch.abs(weight))
        if absmax <= 0:
            return weight
        scale = absmax / q
        return scale * torch.clamp(RoundSTE.apply(weight / scale), -q, q)


class FakeQuantParametrization(nn.Module):
    """Parametrization wrapper that applies fake quant in forward pass."""

    def __init__(self, quantizer, enabled=True):
        super().__init__()
        self.quantizer = quantizer
        self.enabled = bool(enabled)

    def forward(self, weight):
        if not self.enabled:
            return weight
        return self.quantizer(weight)


def _get_fake_quant_parametrization(module):
    """Return the first FakeQuantParametrization on module weight, or None."""
    plist = getattr(getattr(module, "parametrizations", None), "weight", None)
    if plist is None:
        return None
    return next((p for p in plist if isinstance(p, FakeQuantParametrization)), None)


def attach_weight_quantizers(model, bitwidth, exclude_layers=None, enabled=True):
    """Attach or update fake-quant parametrizations on trainable layer weights.

    Only module weights are quantized (not biases). Exclusion is name-based
    and controlled by ``exclude_layers`` tokens.

    Args:
        model: Model to modify in-place.
        bitwidth: Quantization bit-width.
        exclude_layers: Optional list of substrings; matching module names are
            skipped.
        enabled: Whether attached fake quantizers are active.

    Returns:
        Dict with ``attached_modules`` and ``updated_modules`` counts.
    """
    bits = int(bitwidth)
    excludes = tuple(str(item).lower() for item in (exclude_layers or []))
    attached = 0
    updated = 0

    for name, module in model.named_modules():
        if excludes and any(token in name.lower() for token in excludes):
            continue

        fq = _get_fake_quant_parametrization(module)
        if fq is not None:
            fq.quantizer.bitwidth = bits
            fq.enabled = bool(enabled)
            updated += 1
            continue

        weight = getattr(module, "weight", None)
        if (
            not isinstance(weight, nn.Parameter)
            or not weight.requires_grad
            or weight.numel() == 0
        ):
            continue

        if not parametrize.is_parametrized(module, "weight"):
            parametrize.register_parametrization(
                module,
                "weight",
                FakeQuantParametrization(
                    quantizer=UniformSymmetric(bitwidth=bits),
                    enabled=enabled,
                ),
            )
            attached += 1

    return {"attached_modules": int(attached), "updated_modules": int(updated)}


def toggle_quantization(model, enabled):
    """Enable/disable attached fake quantizers."""
    enabled = bool(enabled)
    toggled = 0
    for module in model.modules():
        fq = _get_fake_quant_parametrization(module)
        if fq is not None:
            fq.enabled = enabled
            toggled += 1
    return toggled


def quantize_qat(model, quant_cfg):
    """Prepare fake-quant QAT parametrizations.

    Args:
        model: Model to quantize.
        quant_cfg: Config dict containing at least ``q``.

    Returns:
        Dict describing QAT setup/update state.
    """
    bits = int(quant_cfg.get("q", 32))
    if bits < 2:
        raise ValueError(f"q must be >= 2, got {bits}")
    if bits >= 32:
        disabled = toggle_quantization(model, False)
        return {"qat_enabled": False, "q_bits": bits, "disabled_modules": int(disabled)}

    exclude_layers = quant_cfg.get("exclude_layers", [])
    attach_stats = attach_weight_quantizers(
        model,
        bitwidth=bits,
        exclude_layers=exclude_layers,
        enabled=True,
    )
    attach_stats["qat_enabled"] = True
    attach_stats["q_bits"] = bits
    return attach_stats