"""LightningModule training an MLP that maps scalar index -> block vector."""

import os

import torch
import torch.nn as nn
import lightning as L

from .mlp import MLP
from .siren import SIREN


def encode_indices_bits(indices: torch.Tensor, n_bits: int = 32) -> torch.Tensor:
    """Encode integer indices as fixed-width binary vectors (LSB first, float)."""
    flat = indices.long().reshape(-1, 1)
    bit_pos = torch.arange(n_bits, device=flat.device)
    return ((flat >> bit_pos) & 1).float()


class LitIndexToBlock(L.LightningModule):
    """Regress a 2D weight block from its (binary-encoded) dictionary index."""

    def __init__(
        self,
        block_size: int,
        index_bits: int = 32,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-2,
        epochs: int = 200,
        rel_eps: float = 1e-6,
        arch: str = "mlp",
        hidden_dim: int = 256,
        depth: int = 5,
        w0_first: float = 30.0,
        w0_hidden: float = 1.0,
        save_init_path: str | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        if arch == "mlp":
            self.model = MLP(input_dim=index_bits, num_classes=block_size)
        elif arch == "mlp2":
            self.model = nn.Sequential(
                nn.Linear(index_bits, block_size),
                nn.ReLU(inplace=True),
                nn.Linear(block_size, block_size),
                nn.Tanh(),
            )

        elif arch == "siren":
            self.model = SIREN(
                input_dim=index_bits,
                num_classes=block_size,
                hidden_dim=hidden_dim,
                depth=depth,
                w0_first=w0_first,
                w0_hidden=w0_hidden,
            )
        else:
            raise ValueError(f"Unknown arch={arch!r}; expected 'mlp' or 'siren'")

        if save_init_path:
            os.makedirs(os.path.dirname(save_init_path), exist_ok=True)
            torch.save(self.state_dict(), save_init_path)

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, prefix):
        if len(batch) == 3:
            x, y, w = batch
        else:
            x, y = batch
            w = None
        pred = self(x)
        sq_err = (pred - y).pow(2)
        mse = sq_err.mean().detach()
        if w is not None:
            # Fisher-weighted MSE: penalize errors on high-sensitivity weights.
            loss = (sq_err * w).mean()
        else:
            eps = float(self.hparams.rel_eps)  # type: ignore
            loss = (sq_err / (y.pow(2) + eps)).mean()
        self.log(f"{prefix}/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}/mse", mse, on_step=False, on_epoch=True)
        self.log(f"{prefix}/rmse", mse.sqrt(), on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,  # type: ignore
            weight_decay=self.hparams.weight_decay,  # type: ignore
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.epochs,  # type: ignore
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
