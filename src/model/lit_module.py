"""LightningModule wrapping MLP for classification training."""

import os

import torch
import torch.nn as nn
import lightning as L

from .mlp import MLP


class LitMLP(L.LightningModule):
    """LightningModule for MLP image classification.

    Args:
        input_dim: Flattened input dimension (channels * height * width).
        num_classes: Number of output classes.
        learning_rate: Initial learning rate for AdamW optimizer.
        weight_decay: Weight decay (L2 regularization) factor.
        epochs: Number of epochs (for cosine LR scheduler).
        save_init_path: Optional path to save initial model weights.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 10,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-2,
        epochs: int = 200,
        save_init_path: str | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = MLP(input_dim, num_classes)
        self.criterion = nn.CrossEntropyLoss()

        # Save initial model weights if path is provided
        if save_init_path:
            os.makedirs(os.path.dirname(save_init_path), exist_ok=True)
            torch.save(self.state_dict(), save_init_path)

    def forward(self, x):
        """Run a forward pass through the MLP."""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Compute training loss and log metrics."""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        pred = logits.argmax(dim=1)
        acc = pred.eq(y).float().mean()

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Compute validation loss and log metrics."""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        pred = logits.argmax(dim=1)
        acc = pred.eq(y).float().mean()

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        """Compute test loss and log metrics."""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        pred = logits.argmax(dim=1)
        acc = pred.eq(y).float().mean()

        self.log("test/loss", loss, on_epoch=True)
        self.log("test/acc", acc, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """Configure AdamW optimizer with cosine LR scheduler."""
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
