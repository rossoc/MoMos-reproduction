"""LightningModule wrapping MLP for classification training."""

import os

import torch
import torch.nn as nn
import lightning as L

from .mlp import MLP
from quantizers.momos2d import tensor2D_to_blocks, blocks_to_tensor2D


_MOTIF_FORMAT = "momos2d-motifs-v1"


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

    def save_as_motifs(self, path: str, rows: int, cols: int) -> None:
        """Save the model in compressed motif form.

        Each trainable parameter is split into ``rows x cols`` blocks; identical
        blocks are deduplicated into a global motif library and replaced by
        indices into it. The file stores ``rows``, ``cols``, the motif library,
        per-parameter index tensors, original parameter shapes, any
        non-trainable buffers, and the ``LitMLP`` init kwargs so the full
        module can be rebuilt via :meth:`load_from_motifs`.
        """
        rows, cols = int(rows), int(cols)

        all_blocks: list[torch.Tensor] = []
        param_specs: list[tuple[str, tuple, int]] = []
        for name, param in self.model.named_parameters():
            if not (param.requires_grad and param.numel() > 0):
                continue
            blocks, _, _ = tensor2D_to_blocks(param.detach().cpu(), rows, cols)
            param_specs.append((name, tuple(param.shape), int(blocks.size(0))))
            all_blocks.append(blocks)

        if all_blocks:
            motifs, inverse = torch.unique(
                torch.cat(all_blocks, dim=0), dim=0, return_inverse=True
            )
        else:
            motifs = torch.empty(0, rows * cols)
            inverse = torch.empty(0, dtype=torch.long)

        n_motifs = int(motifs.size(0))
        if n_motifs <= 256:
            idx_dtype = torch.uint8
        elif n_motifs <= 32768:
            idx_dtype = torch.int16
        else:
            idx_dtype = torch.int32
        inverse = inverse.to(idx_dtype)

        param_indices: dict[str, torch.Tensor] = {}
        offset = 0
        for name, _, n_blocks in param_specs:
            param_indices[name] = inverse[offset : offset + n_blocks].clone()
            offset += n_blocks

        trainable_names = {name for name, _, _ in param_specs}
        extra_state = {
            k: v.detach().cpu()
            for k, v in self.model.state_dict().items()
            if k not in trainable_names
        }

        payload = {
            "format": _MOTIF_FORMAT,
            "rows": rows,
            "cols": cols,
            "motifs": motifs,
            "param_indices": param_indices,
            "param_shapes": {n: s for n, s, _ in param_specs},
            "extra_state": extra_state,
            "init_kwargs": {
                "input_dim": int(self.hparams.input_dim),  # type: ignore
                "num_classes": int(self.hparams.num_classes),  # type: ignore
                "learning_rate": float(self.hparams.learning_rate),  # type: ignore
                "weight_decay": float(self.hparams.weight_decay),  # type: ignore
                "epochs": int(self.hparams.epochs),  # type: ignore
            },
        }
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        torch.save(payload, path)

    @classmethod
    def load_from_motifs(cls, path: str) -> "LitMLP":
        """Rebuild a :class:`LitMLP` from a checkpoint produced by
        :meth:`save_as_motifs`.

        Reads ``rows``/``cols`` and all metadata from the file, materialises
        each parameter by gathering its blocks from the motif library, and
        restores any non-trainable buffers.
        """
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("format") != _MOTIF_FORMAT:
            raise ValueError(
                f"Unrecognized motif checkpoint format: {payload.get('format')!r}"
            )

        rows = int(payload["rows"])
        cols = int(payload["cols"])
        motifs = payload["motifs"]

        module = cls(**payload["init_kwargs"])

        state = dict(module.model.state_dict())
        for name, idx in payload["param_indices"].items():
            true_shape = tuple(payload["param_shapes"][name])
            block_shape = true_shape if len(true_shape) >= 2 else (1, true_shape[0])
            blocks = motifs[idx.long()]
            state[name] = blocks_to_tensor2D(blocks, block_shape, rows, cols).reshape(
                true_shape
            )
        for name, tensor in payload["extra_state"].items():
            state[name] = tensor
        module.model.load_state_dict(state)
        return module

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
