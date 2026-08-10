"""Unit tests for LitClassifier's optimizer dispatch.

Guards against regressing to a single hardcoded optimizer: different backbones
need different optimizer profiles (e.g. small CIFAR ResNets are trained with
SGD+momentum, not AdamW) — mirrors the upstream `saintslab/MoMos` per-model
optimizer profiles.
"""

import pytest
import torch

from model.mlp import MLP
from model.lit_module import LitClassifier


def _make_module(**kwargs):
    backbone = MLP(input_dim=32 * 32 * 3, num_classes=10)
    return LitClassifier(backbone=backbone, num_classes=10, **kwargs)


def test_configure_optimizers_defaults_to_adamw():
    module = _make_module()
    result = module.configure_optimizers()
    assert isinstance(result["optimizer"], torch.optim.AdamW)


def test_configure_optimizers_adamw_uses_configured_lr_and_weight_decay():
    module = _make_module(learning_rate=3e-4, weight_decay=1e-2, optimizer="adamw")
    optimizer = module.configure_optimizers()["optimizer"]
    group = optimizer.param_groups[0]
    assert group["lr"] == pytest.approx(3e-4)
    assert group["weight_decay"] == pytest.approx(1e-2)


def test_configure_optimizers_sgd_uses_momentum_and_weight_decay():
    module = _make_module(
        optimizer="sgd", learning_rate=0.1, weight_decay=1e-4, momentum=0.9
    )
    optimizer = module.configure_optimizers()["optimizer"]
    assert isinstance(optimizer, torch.optim.SGD)
    group = optimizer.param_groups[0]
    assert group["lr"] == pytest.approx(0.1)
    assert group["weight_decay"] == pytest.approx(1e-4)
    assert group["momentum"] == pytest.approx(0.9)


def test_configure_optimizers_rejects_unknown_optimizer():
    module = _make_module(optimizer="rmsprop")
    with pytest.raises(ValueError):
        module.configure_optimizers()


def test_configure_optimizers_scheduler_uses_configured_epochs():
    module = _make_module(optimizer="sgd", epochs=42)
    scheduler = module.configure_optimizers()["lr_scheduler"]["scheduler"]
    assert scheduler.T_max == 42
