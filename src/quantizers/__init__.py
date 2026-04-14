"""MoMos quantization package."""

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
    METHODS,
    available_methods,
    MoMos,
    quantize,
)

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
    "MoMos",
    "quantize",
]
