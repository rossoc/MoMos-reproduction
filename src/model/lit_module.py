"""LightningModule wrapping MLP for classification training."""

import torch
import torch.nn as nn
import lightning as L

from model.mlp import MLP


class LitMLP(L.LightningModule):
    """LightningModule for MLP image classification.

    Args:
        input_dim: Flattened input dimension (channels * height * width).
        num_classes: Number of output classes.
        learning_rate: Initial learning rate for AdamW optimizer.
        weight_decay: Weight decay (L2 regularization) factor.
        epochs: Number of epochs (for cosine LR scheduler).
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 10,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-2,
        epochs: int = 200,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = MLP(input_dim, num_classes)
        self.criterion = nn.CrossEntropyLoss()

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
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.epochs,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
