"""SIREN: sinusoidal-activation MLP for high-frequency regression.

Reference: Sitzmann et al., "Implicit Neural Representations with Periodic
Activation Functions" (NeurIPS 2020).
"""

import math

import torch
import torch.nn as nn


class SineLayer(nn.Module):
    """Linear -> sin(w0 * x) with SIREN initialization."""

    def __init__(self, in_features: int, out_features: int, w0: float, is_first: bool):
        super().__init__()
        self.w0 = float(w0)
        self.linear = nn.Linear(in_features, out_features)
        with torch.no_grad():
            if is_first:
                bound = 1.0 / in_features
            else:
                bound = math.sqrt(6.0 / in_features) / self.w0
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * self.linear(x))


class SIREN(nn.Module):
    """SIREN MLP: ``depth`` sine layers + a final linear head."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        depth: int = 5,
        w0_first: float = 30.0,
        w0_hidden: float = 1.0,
    ):
        super().__init__()
        assert depth >= 1
        layers: list[nn.Module] = [
            SineLayer(input_dim, hidden_dim, w0=w0_first, is_first=True)
        ]
        for _ in range(depth - 1):
            layers.append(
                SineLayer(hidden_dim, hidden_dim, w0=w0_hidden, is_first=False)
            )
        self.body = nn.Sequential(*layers)

        head = nn.Linear(hidden_dim, num_classes)
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden_dim) / w0_hidden
            head.weight.uniform_(-bound, bound)
            head.bias.zero_()
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1)
        return self.head(self.body(x))
