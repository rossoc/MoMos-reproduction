"""Lightning module: train the Mamba hypernetwork to *propose* a fresh MLP.

Every step is a single end-to-end backprop:

    logits_K = hypernet(...)               # [K, nbh, nbw] per layer
    mix      = softmax(logits_K, dim=0)    # convex combination over K motifs
    W_l      = mix @ motifs                # blended weight tensor per layer
    y_hat    = target_mlp(x; W)            # via torch.func.functional_call
    loss     = CE(y_hat, y)
    loss.backward()                        # flows through W -> softmax -> hypernet

No temperature, no Gumbel sampling, no STE, no macro-batching. Memory is
controlled by ``motif_chunk_size`` (motifs forwarded per Mamba call) and
``mamba_chunk_size`` (SSD chunk inside Mamba2). When grad is enabled and
``motif_chunk_size < K``, each chunk's Mamba forward is gradient-checkpointed
so only one chunk's activations are alive at a time.
"""

import torch
import torch.nn.functional as F
import lightning as L
from torch.func import functional_call
from torch.utils.checkpoint import checkpoint

from .mamba import Mamba
from .lit_module import LitMLP
from quantizers.momos2d import blocks_to_tensor2D


class LitStructuredMomos(L.LightningModule):
    def __init__(
        self,
        # --- Mamba (hypernet) config ---
        n_motifs: int,
        num_heads: int,
        head_dim: int,
        hidden_size: int,
        num_rows: int,
        num_cols: int,
        num_layers: int,
        state_size: int,
        num_hidden_layers: int,
        n_groups: int,
        hidden_act: str,
        time_step_min: float,
        time_step_max: float,
        out_channels: int,
        output_layers: int,
        # --- Motif library + layer geometry (from MotifMaskDataset) ---
        motifs: torch.Tensor,
        layer_shapes: list,
        motif_rows: int,
        motif_cols: int,
        rows: int,
        cols: int,
        n_rows_per_layer: list,
        n_cols_per_layer: list,
        # --- Target MLP geometry ---
        mlp_input_dim: int,
        mlp_num_classes: int,
        # --- Optimization ---
        learning_rate: float = 3e-3,
        weight_decay: float = 1e-2,
        epochs: int = 200,
        # --- Memory ---
        motif_chunk_size: int | None = None,
        mamba_chunk_size: int | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["motifs", "layer_shapes"])

        self.hypernet = Mamba(
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
            out_channels=out_channels,
            output_layers=output_layers,
            mamba_chunk_size=mamba_chunk_size,
        )

        # Target MLP — a structural shell. Its parameters are replaced per step
        # by hypernet-generated tensors via ``functional_call``, so we freeze
        # them (functional_call uses the dict values regardless).
        target = LitMLP(input_dim=mlp_input_dim, num_classes=mlp_num_classes).model
        for p in target.parameters():
            p.requires_grad_(False)
        self.target = target
        self._target_param_names: list[str] = [
            name for name, _ in target.named_parameters()
        ]

        self.layer_shapes = list(layer_shapes)
        self.motif_rows = int(motif_rows)
        self.motif_cols = int(motif_cols)
        self.rows = int(rows)
        self.cols = int(cols)
        self.n_rows_per_layer = n_rows_per_layer
        self.n_cols_per_layer = n_cols_per_layer

        self.K = int(motifs.shape[0])
        self.D = self.motif_rows * self.motif_cols
        self.motif_chunk_size = motif_chunk_size
        self.best_val_acc = 0.0

        self.register_buffer(
            "motifs", motifs.reshape(self.K, self.D).detach().clone(), persistent=False
        )

        if len(self._target_param_names) != len(self.layer_shapes):
            raise ValueError(
                f"Target MLP exposes {len(self._target_param_names)} params but "
                f"{len(self.layer_shapes)} layer_shapes were given — the target "
                f"architecture must match the one used to derive the motif library."
            )

    # ---------------------------------------------------------- weight build
    def _layer_block_counts(self, layer_idx: int) -> tuple[int, int]:
        shape = self.layer_shapes[layer_idx]
        orig_h, orig_w = int(shape[-2]), int(shape[-1])
        nbh = (orig_h + self.motif_rows - 1) // self.motif_rows
        nbw = (orig_w + self.motif_cols - 1) // self.motif_cols
        return nbh, nbw

    def _layer_logits(self, layer_idx: int) -> torch.Tensor:
        """``[K, nbh, nbw]`` pre-softmax logits from the hypernet for one layer.

        Motifs are forwarded in chunks of ``motif_chunk_size`` and each chunk's
        Mamba pass is gradient-checkpointed during training, so only one chunk's
        activations are alive at a time.
        """
        device = self.device
        n_rows = self.n_rows_per_layer[layer_idx]
        n_cols = self.n_cols_per_layer[layer_idx]
        layer_id = torch.tensor(layer_idx, device=device, dtype=torch.long)
        nbh, nbw = self._layer_block_counts(layer_idx)
        H_pad = n_rows * self.rows
        W_pad = n_cols * self.cols

        chunk = self.motif_chunk_size or self.K
        chunk = max(1, min(int(chunk), self.K))
        use_ckpt = torch.is_grad_enabled() and chunk < self.K

        def _run(ids: torch.Tensor) -> torch.Tensor:
            out = self.hypernet(ids, layer_id, n_rows, n_cols)
            c = out.shape[0]
            out = out.view(c, n_rows, n_cols, self.rows, self.cols)
            out = out.permute(0, 1, 3, 2, 4).reshape(c, H_pad, W_pad)
            return out[:, :nbh, :nbw].contiguous()

        all_ids = torch.arange(self.K, device=device, dtype=torch.long)
        pieces = []
        for start in range(0, self.K, chunk):
            ids = all_ids[start : start + chunk]
            if use_ckpt:
                pieces.append(checkpoint(_run, ids, use_reentrant=False))
            else:
                pieces.append(_run(ids))
        return torch.cat(pieces, dim=0)

    def _build_weight_dict(self) -> dict[str, torch.Tensor]:
        """For each target-MLP param: ``W = softmax(logits) @ motifs``, tiled."""
        weight_dict: dict[str, torch.Tensor] = {}
        for L_idx, name in enumerate(self._target_param_names):
            logits = self._layer_logits(L_idx)                  # [K, nbh, nbw]
            mix = F.softmax(logits, dim=0)                      # [K, nbh, nbw]
            blocks = torch.einsum("kd,khw->hwd", self.motifs, mix)  # [nbh, nbw, D]
            nbh, nbw = self._layer_block_counts(L_idx)
            blocks = blocks.view(nbh * nbw, self.D)
            weight_dict[name] = blocks_to_tensor2D(
                blocks, self.layer_shapes[L_idx], self.motif_rows, self.motif_cols
            )
        return weight_dict

    # ---------------------------------------------------------- lightning
    def _shared_step(self, batch, prefix: str) -> torch.Tensor:
        x, y = batch
        W = self._build_weight_dict()
        logits = functional_call(self.target, W, (x,))
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log(f"{prefix}/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, "val")
        return loss

    def on_validation_epoch_end(self):
        val_acc = self.trainer.callback_metrics.get("val/acc")
        if val_acc is not None and float(val_acc) > self.best_val_acc:
            self.best_val_acc = float(val_acc)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.hypernet.parameters(),
            lr=self.hparams.learning_rate,  # type: ignore[attr-defined]
            weight_decay=self.hparams.weight_decay,  # type: ignore[attr-defined]
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.hparams.epochs  # type: ignore[attr-defined]
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "epoch", "frequency": 1},
        }
