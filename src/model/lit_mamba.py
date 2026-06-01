import os

import torch
import torch.nn as nn
import lightning as L

from .mamba import Mamba
from .lit_module import LitMLP
from quantizers.momos2d import blocks_to_tensor2D
from quantizers.block_utils import iter_trainable_params


class LitMamba(L.LightningModule):
    def __init__(
        self,
        n_motifs,
        num_heads,
        head_dim,
        hidden_size,
        num_rows,
        num_cols,
        num_layers,
        state_size,
        num_hidden_layers,
        n_groups,
        hidden_act,
        time_step_min,
        time_step_max,
        cnn_out_channels,
        cnn_kernel_size,
        cnn_padding,
        learning_rate: float = 3e-3,
        weight_decay: float = 1e-2,
        epochs: int = 200,
        save_init_path: str | None = None,
        # Reconstruction + validation (optional — used by train_hypernet.py)
        motifs: torch.Tensor | None = None,
        layer_shapes: list | None = None,
        motif_rows: int | None = None,
        motif_cols: int | None = None,
        rows: int | None = None,
        cols: int | None = None,
        n_rows_per_layer: list | None = None,
        n_cols_per_layer: list | None = None,
        image_datamodule=None,
        mlp_input_dim: int | None = None,
        mlp_num_classes: int | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["motifs", "layer_shapes", "image_datamodule"]
        )

        self.model = Mamba(
            n_motifs=n_motifs,
            num_heads=num_heads,
            head_dim=head_dim,
            hidden_size=hidden_size,
            num_rows=num_rows,
            num_cols=num_cols,
            num_layers=num_layers,
            state_size=state_size,
            num_hidden_layers=num_hidden_layers,
            n_groups=n_groups,
            hidden_act=hidden_act,
            time_step_min=time_step_min,
            time_step_max=time_step_max,
            cnn_out_channels=cnn_out_channels,
            cnn_kernel_size=cnn_kernel_size,
            cnn_padding=cnn_padding,
        )
        self.criterion = nn.BCEWithLogitsLoss()

        if motifs is not None:
            self.register_buffer("motifs", motifs.detach().clone(), persistent=False)
        else:
            self.motifs = None

        self.layer_shapes = layer_shapes
        self.motif_rows = motif_rows
        self.motif_cols = motif_cols
        self.rows = rows
        self.cols = cols
        self.n_rows_per_layer = n_rows_per_layer
        self.n_cols_per_layer = n_cols_per_layer
        self.image_datamodule = image_datamodule
        self.mlp_input_dim = mlp_input_dim
        self.mlp_num_classes = mlp_num_classes

        if save_init_path:
            os.makedirs(os.path.dirname(save_init_path), exist_ok=True)
            torch.save(self.state_dict(), save_init_path)

    def forward(self, x):
        if len(x) == 4:
            return self.model(x[0], x[1], x[2], x[3])
        elif len(x) == 6:
            return self.model(x[0], x[1], x[2], x[3], x[4], x[5])
        else:
            raise ValueError(
                "Wrong input to the model: expected length size of 4 or 6, found length",
                len(x),
                "corresponding to: ",
                x,
            )

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y.float())

        pred = logits > 0
        acc = pred.eq(y.bool()).float().mean()

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def build_mlp(self, motifs: torch.Tensor | None = None) -> LitMLP:
        """Assemble a `LitMLP` whose weights come from Mamba-predicted masks.

        For every (motif, layer) the Mamba head produces logits over
        ``rows*cols`` subblock positions; the per-layer motif assignment is the
        argmax across motifs at each fine position, and each layer's weight
        tensor is rebuilt by tiling ``motifs[predicted_index]`` back into 2D.
        """
        motifs_t = motifs if motifs is not None else self.motifs
        if motifs_t is None:
            raise ValueError("build_mlp requires `motifs` (constructor arg or kwarg).")
        if self.mlp_input_dim is None or self.mlp_num_classes is None:
            raise ValueError(
                "build_mlp requires `mlp_input_dim` and `mlp_num_classes`."
            )

        device = self.device
        K = int(motifs_t.shape[0])
        motifs_t = motifs_t.to(device)

        lit_mlp = LitMLP(
            input_dim=self.mlp_input_dim,
            num_classes=self.mlp_num_classes,
        ).to(device)

        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                params = list(iter_trainable_params(lit_mlp.model))
                for L_idx, param in enumerate(params):
                    layer_h = int(self.n_rows_per_layer[L_idx])
                    layer_w = int(self.n_cols_per_layer[L_idx])
                    H_pad = layer_h * int(self.rows)
                    W_pad = layer_w * int(self.cols)

                    logits_full = torch.empty(K, H_pad, W_pad, device=device)
                    for k in range(K):
                        motif_id = torch.tensor(k, device=device, dtype=torch.long)
                        layer_id = torch.tensor(L_idx, device=device, dtype=torch.long)
                        out = self.model(motif_id, layer_id, layer_h, layer_w)
                        out = out.view(layer_h, layer_w, int(self.rows), int(self.cols))
                        out = out.permute(0, 2, 1, 3).reshape(H_pad, W_pad)
                        logits_full[k] = out

                    predicted_M_padded = logits_full.argmax(dim=0)

                    shape = self.layer_shapes[L_idx]
                    orig_h, orig_w = int(shape[-2]), int(shape[-1])
                    num_blocks_h = (orig_h + int(self.motif_rows) - 1) // int(self.motif_rows)
                    num_blocks_w = (orig_w + int(self.motif_cols) - 1) // int(self.motif_cols)
                    predicted_M = predicted_M_padded[:num_blocks_h, :num_blocks_w]

                    blocks = motifs_t[predicted_M]
                    blocks = blocks.reshape(
                        num_blocks_h * num_blocks_w,
                        int(self.motif_rows) * int(self.motif_cols),
                    )
                    reconstructed = blocks_to_tensor2D(
                        blocks, shape, int(self.motif_rows), int(self.motif_cols)
                    )
                    param.data.copy_(reconstructed.view_as(param).to(param.dtype))
        finally:
            if was_training:
                self.train()

        return lit_mlp

    def on_train_epoch_end(self):
        """Reconstruct an MLP from predicted masks and evaluate on the validation fold."""
        if self.image_datamodule is None or self.motifs is None:
            return

        val_loader = self.image_datamodule.val_dataloader()
        if val_loader is None:
            return

        lit_mlp = self.build_mlp()
        lit_mlp.eval()
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        total_correct = 0.0
        n = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                logits = lit_mlp(x)
                loss = criterion(logits, y)
                bs = x.size(0)
                total_loss += loss.item() * bs
                total_correct += logits.argmax(dim=1).eq(y).float().sum().item()
                n += bs

        if n == 0:
            return

        self.log("val/loss", total_loss / n, prog_bar=True)
        self.log("val/acc", total_correct / n, prog_bar=True)

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
