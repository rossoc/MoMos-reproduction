"""LightningModule training a Mamba-based hypernet to regress MLP weights end-to-end.

Each training step:
  1. Calls the hypernet once per MLP parameter to materialize a fresh weight tensor.
  2. Forwards the MLP on the input batch using ``torch.func.functional_call`` with
     the hypernet-produced tensors as parameters.
  3. Backprops the classification loss through the functional call into the
     hypernet — no discrete ops break the graph.
"""

import torch
import torch.nn as nn
import lightning as L
from torch.func import functional_call

from .mamba import MambaRegressor
from .mlp import MLP
from quantizers.momos2d import blocks_to_tensor2D


class LitMambaRegressor(L.LightningModule):
    def __init__(
        self,
        mlp_input_dim: int,
        mlp_num_classes: int,
        rows: int,
        cols: int,
        num_heads: int,
        head_dim: int,
        hidden_size: int,
        state_size: int,
        num_hidden_layers: int,
        n_groups: int,
        hidden_act: str,
        time_step_min: float,
        time_step_max: float,
        learning_rate: float = 3e-3,
        weight_decay: float = 1e-2,
        epochs: int = 200,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.rows = int(rows)
        self.cols = int(cols)
        self.criterion = nn.CrossEntropyLoss()

        # Template MLP — we never train its parameters; it just provides the
        # module structure for ``functional_call``.
        self.template_mlp = MLP(mlp_input_dim, mlp_num_classes)
        for p in self.template_mlp.parameters():
            p.requires_grad_(False)

        # Per-parameter spec: (name, shape_2d, layer_h, layer_w, orig_shape)
        self.param_specs = []
        max_h = 1
        max_w = 1
        for name, p in self.template_mlp.named_parameters():
            orig_shape = tuple(p.shape)
            if p.dim() == 1:
                shape_2d = (1, orig_shape[0])
            elif p.dim() == 2:
                shape_2d = (orig_shape[0], orig_shape[1])
            else:
                raise ValueError(
                    f"Unsupported parameter rank for {name}: shape={orig_shape}"
                )
            h, w = shape_2d
            layer_h = (h + self.rows - 1) // self.rows
            layer_w = (w + self.cols - 1) // self.cols
            self.param_specs.append((name, shape_2d, layer_h, layer_w, orig_shape))
            max_h = max(max_h, layer_h)
            max_w = max(max_w, layer_w)

        num_layers = len(self.param_specs)
        out_channels = self.rows * self.cols

        self.model = MambaRegressor(
            num_heads=num_heads,
            head_dim=head_dim,
            hidden_size=hidden_size,
            num_rows=max_h,
            num_cols=max_w,
            num_layers=num_layers,
            state_size=state_size,
            num_hidden_layers=num_hidden_layers,
            n_groups=n_groups,
            hidden_act=hidden_act,
            time_step_min=time_step_min,
            time_step_max=time_step_max,
            out_channels=out_channels,
        )

    def build_params(self) -> dict[str, torch.Tensor]:
        """Predict every MLP parameter from the hypernet (differentiable)."""
        device = self.device
        params: dict[str, torch.Tensor] = {}
        for L_idx, (name, shape_2d, layer_h, layer_w, orig_shape) in enumerate(
            self.param_specs
        ):
            layer_id = torch.tensor(L_idx, device=device, dtype=torch.long)
            out = self.model(layer_id, layer_h, layer_w)  # [layer_h*layer_w, rows*cols]
            W2d = blocks_to_tensor2D(out, shape_2d, self.rows, self.cols)
            if len(orig_shape) == 1:
                W = W2d.reshape(orig_shape[0])
            else:
                W = W2d
            params[name] = W
        return params

    def forward(self, x):
        params = self.build_params()
        return functional_call(self.template_mlp, params, (x,))

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = logits.argmax(dim=1).eq(y).float().mean()
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = logits.argmax(dim=1).eq(y).float().mean()
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        trainable = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable,
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
