"""MoMos quantization package."""

import time

from .fake_quant import (
    RoundSTE,
    UniformSymmetric,
    FakeQuantParametrization,
    attach_weight_quantizers,
    toggle_quantization,
    quantize_qat,
)

from .block_utils import (
    iter_trainable_params,
    tensor_to_blocks,
    blocks_to_tensor,
    count_total_blocks,
    k_from_capacity,
)

from .momos import (
    momos,
    quantize_momos,
)

from .momos2d import quantize_momos2D, momos2D

__all__ = [
    # Fake quantization
    "RoundSTE",
    "UniformSymmetric",
    "FakeQuantParametrization",
    "attach_weight_quantizers",
    "toggle_quantization",
    "quantize_qat",
    # Block utilities
    "iter_trainable_params",
    "tensor_to_blocks",
    "blocks_to_tensor",
    "count_total_blocks",
    "k_from_capacity",
    # MoMos core
    "momos",
    "quantize_momos",
    "METHODS",
    "available_methods",
    "quantize",
    # MoMos2D core
    "momos2D",
    "quantize_momos2D",
]

METHODS = {
    "qat": quantize_qat,
    "momos": quantize_momos,
    "momos2d": quantize_momos2D,
}


def available_methods():
    """Return supported quantization method names."""
    return sorted(METHODS.keys())


def quantize(model, quant_cfg):
    """Dispatch quantization by method and attach elapsed time.

    Args:
        model: Model to quantize.
        quant_cfg: Quantization config dict with ``method`` key.

    Returns:
        Stats dict with ``method`` and ``q_time``, or ``None``.
    """
    if not quant_cfg:
        return {}

    start = time.perf_counter()
    method = str(quant_cfg.get("method", "qat")).lower()
    fn = METHODS.get(method)
    if fn is None:
        raise ValueError(
            f"Unsupported quantization method: {method}. Available: {', '.join(available_methods())}"
        )
    out = fn(model, quant_cfg) or {}
    out["method"] = method
    out["q_time"] = time.perf_counter() - start
    return out
