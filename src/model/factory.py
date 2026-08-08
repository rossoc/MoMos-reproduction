"""Backbone factory mapping a model config to an ``nn.Module``.

Keeps model selection in one place so ``train.py`` stays architecture-agnostic.
Dispatch mirrors the upstream ``saintslab/MoMs`` ``get_model``:

  * ``mlp``          -> local :class:`MLP` sized to ``img_size``
  * ``resnet*``      -> local CIFAR ResNet implementation in ``resnet_s``
                       (Idelbayev's proper CIFAR variant; ResNet-20 = 0.27M params)
  * anything else    -> delegated to ``timm`` (e.g. ``tiny_vit_5m_224``)

timm models like TinyViT do not accept an ``img_size`` kwarg, so they are
built at their native resolution and the datamodule is responsible for
resizing inputs to match (the model config's ``img_size`` override drives
that resize in ``train.py``).
"""

import inspect

import torch.nn as nn

from .mlp import MLP
from . import resnet_s


def build_local_resnet(name, num_classes, in_channels=3):
    """Build a local ResNet implementation when ``name`` matches.

    Mirrors the upstream ``build_local_resnet`` dispatch: only names starting
    with ``resnet`` are handled, and the builder calls the matching function in
    ``resnet_s`` with ``num_classes`` / ``in_channels`` when it accepts them.

    Args:
        name: Requested model name (e.g. ``resnet20``).
        num_classes: Number of output classes.
        in_channels: Number of input image channels.

    Returns:
        Local ResNet instance when available, else ``None``.
    """
    if not str(name).startswith("resnet"):
        return None
    fn = getattr(resnet_s, name, None)
    if not callable(fn):
        return None
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn()
    kwargs = {}
    if "num_classes" in sig.parameters:
        kwargs["num_classes"] = num_classes
    if "in_channels" in sig.parameters:
        kwargs["in_channels"] = int(in_channels)
    return fn(**kwargs)


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

    local_resnet = build_local_resnet(name, num_classes, in_channels=in_channels)
    if local_resnet is not None:
        return local_resnet

    import timm

    return timm.create_model(
        model_cfg.model_id,
        num_classes=num_classes,
        in_chans=int(in_channels),
        pretrained=False,
    )
