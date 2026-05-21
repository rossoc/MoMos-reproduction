"""LightningModule mapping a block-index to its MoMos motif id (classification).

When the source model has been MoMos-quantized with a small capacity (few
motifs), each weight block is assigned to one of K << n_blocks centroids.
Learning the assignment index -> motif_id is a discrete classification problem
that is typically much easier than regressing the full block vector.
"""

import os

import torch
import torch.nn as nn
import lightning as L

from .lit_index_to_block import encode_indices_bits
from .siren import SIREN


class LitIndexToMotif(L.LightningModule):
    """One-hidden-layer MLP: binary-encoded index -> motif-id logits."""

    def __init__(
        self,
        num_motifs: int,
        input_dim: int,
        hidden_dim: int = 256,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-2,
        epochs: int = 200,
        arch: str = "mlp",  # 'mlp' (ReLU, 2 hidden) or 'siren'
        depth: int = 5,  # siren only
        w0_first: float = 30.0,  # siren only
        w0_hidden: float = 1.0,  # siren only
        save_init_path: str | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        in_dim = int(input_dim)

        if arch == "siren":
            self.model = SIREN(
                input_dim=in_dim,
                num_classes=num_motifs,
                hidden_dim=hidden_dim,
                depth=depth,
                w0_first=w0_first,
                w0_hidden=w0_hidden,
            )
        elif arch == "mlp":
            self.model = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, num_motifs),
            )
        else:
            raise ValueError(f"Unknown arch={arch!r}; expected 'mlp' or 'siren'")

        if save_init_path:
            os.makedirs(os.path.dirname(save_init_path), exist_ok=True)
            torch.save(self.state_dict(), save_init_path)

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, prefix):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y.long())
        acc = (logits.argmax(dim=-1) == y).float().mean().detach()
        self.log(f"{prefix}/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

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
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


__all__ = ["LitIndexToMotif", "encode_indices_bits"]
