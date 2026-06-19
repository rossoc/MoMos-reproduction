"""Backbone factory mapping a model config to an ``nn.Module``.

Keeps model selection in one place so ``train.py`` stays architecture-agnostic.
``mlp`` builds the local :class:`MLP` sized to ``img_size``; any other name is
delegated to ``timm`` (e.g. ``tiny_vit_5m_224``). timm models like TinyViT do
not accept an ``img_size`` kwarg, so they are built at their native resolution
and the datamodule is responsible for resizing inputs to match (the model
config's ``img_size`` override drives that resize in ``train.py``).
"""

import torch.nn as nn

from .mlp import MLP


def build_backbone(model_cfg, *, in_channels: int, img_size: int, num_classes: int) -> nn.Module:
    """Build a classification backbone from a Hydra model config.

    Args:
        model_cfg: Model config group (has ``name``; for timm models also
            ``model_id``).
        in_channels: Input image channels (e.g. 3 for CIFAR, 1 for MNIST).
        img_size: Effective input side length. Used to size the MLP input; timm
            models are built at their own native resolution (inputs are resized
            to it by the datamodule).
        num_classes: Number of output classes.

    Returns:
        An ``nn.Module`` backbone (used as ``LitClassifier.model``).
    """
    name = str(model_cfg.name)

    if name == "mlp":
        return MLP(int(in_channels) * int(img_size) * int(img_size), num_classes)

    import timm

    return timm.create_model(
        model_cfg.model_id,
        num_classes=num_classes,
        in_chans=int(in_channels),
        pretrained=False,
    )
