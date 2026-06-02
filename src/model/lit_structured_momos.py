"""Lightning module: train the Mamba hypernetwork to *propose* a fresh MLP.

Departure from ``LitMamba``: the hypernet no longer regresses the original
mask grid. Instead, at every training step it emits a categorical distribution
over motifs at each spatial position; Gumbel-softmax (straight-through) turns
those distributions into a near-one-hot, differentiable motif selection; the
selected motifs are tiled back into MLP weight tensors; the resulting MLP is
called on a task batch via ``torch.func.functional_call``; classification CE
flows back through the weights → masks → hypernet.

Regularization pushes the mask distribution toward a true one-hot:

* **Entropy penalty** on the per-position softmax — annealed up over training.
* **Load-balance KL** between the average motif usage and a uniform prior — a
  small constant — to discourage collapse onto one or two motifs.

The Gumbel temperature ``tau`` is annealed linearly from ``tau_start`` to
``tau_end`` over the configured number of epochs.

Memory: per-layer Mamba forwards over all ``K`` motifs are split into chunks
of ``motif_chunk_size`` and wrapped in ``torch.utils.checkpoint`` so only one
chunk's activations are alive at a time. The full ``[K, H_pad, W_pad]`` logit
tensor is materialized once, but motif activations dominate, so chunking is
where memory is actually won.
"""

import torch
import torch.nn.functional as F
import lightning as L
from torch.func import functional_call

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
        motifs: torch.Tensor,  # [K, motif_rows, motif_cols]
        layer_shapes: list,  # per-param original shapes
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
        grad_clip: float = 1.0,
        # --- Gumbel + regularization ---
        tau_start: float = 1.0,
        tau_end: float = 0.1,
        entropy_weight_start: float = 0.0,
        entropy_weight_end: float = 1e-2,
        load_balance_weight: float = 1e-3,
        hard: bool = True,
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

        # Target MLP: a structural shell. Its parameters are *replaced* per step
        # by hypernet-generated tensors via ``functional_call`` — we never train
        # them directly, so freeze.
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

    # --------------------- schedules ---------------------------------------
    def _progress(self) -> float:
        """Linear epoch progress in ``[0, 1]``."""
        max_e = self.trainer.max_epochs if self.trainer is not None else None
        if not max_e or max_e <= 1:
            return 0.0
        return min(1.0, self.current_epoch / (max_e - 1))

    def _tau(self) -> float:
        h = self.hparams
        return float(h.tau_start + (h.tau_end - h.tau_start) * self._progress())  # type: ignore[attr-defined]

    # --------------------- hypernet forward --------------------------------
    def _layer_logits(self, layer_idx: int) -> torch.Tensor:
        """Return ``[K, nbh, nbw]`` pre-softmax logits for one target MLP layer.

        ``logits[k, i, j]`` is the unnormalized score for placing motif ``k`` at
        block position ``(i, j)`` of the layer's block grid. The hypernet emits a
        padded grid of size ``n_rows_per_layer[layer_idx] * self.rows`` by
        ``n_cols_per_layer[layer_idx] * self.cols``; this function crops it to the
        true ``(nbh, nbw)`` derived from ``layer_shapes[layer_idx]``.
        """
        device = self.device
        n_rows = self.n_rows_per_layer[layer_idx]
        n_cols = self.n_cols_per_layer[layer_idx]
        layer_id = torch.tensor(layer_idx, device=device, dtype=torch.long)
        all_ids = torch.arange(self.K, device=device, dtype=torch.long)

        out = self.hypernet(all_ids, layer_id, n_rows, n_cols)
        c = out.shape[0]
        out = out.view(c, n_rows, n_cols, self.rows, self.cols)

        H_pad = n_rows * self.rows
        W_pad = n_cols * self.cols
        logits = out.permute(0, 1, 3, 2, 4).reshape(c, H_pad, W_pad)

        nbh, nbw = self._layer_block_counts(layer_idx)
        logits = logits[:, :nbh, :nbw]  # crop pad
        return logits

    # --------------------- weight generation -------------------------------
    def _layer_block_counts(self, layer_idx: int) -> tuple[int, int]:
        shape = self.layer_shapes[layer_idx]
        orig_h, orig_w = int(shape[-2]), int(shape[-1])
        nbh = (orig_h + self.motif_rows - 1) // self.motif_rows
        nbw = (orig_w + self.motif_cols - 1) // self.motif_cols
        return nbh, nbw

    def _generate_weights(self, hard: bool) -> dict[str, torch.Tensor]:
        """Build the ``functional_call`` weight dict.

        Args:
            hard: if True, use Gumbel-softmax STE (one-hot forward, soft
                gradient); if False, use plain softmax (validation).

        Returns:
            weight_dict
        """
        tau = self._tau()
        weight_dict: dict[str, torch.Tensor] = {}

        for L_idx, name in enumerate(self._target_param_names):
            logits = self._layer_logits(L_idx)  # [K, nbh, nbw]

            if hard:
                weights = F.gumbel_softmax(logits, tau=tau, hard=True, dim=0)
            else:
                weights = F.softmax(logits, dim=0)

            # Z @ M -> W: [nbh, nbw, D]
            blocks = torch.einsum("kd,khw->hwd", self.motifs, weights)
            blocks = blocks.view(blocks.shape[0] * blocks.shape[1], -1)

            weight_dict[name] = blocks_to_tensor2D(
                blocks, self.layer_shapes[L_idx], self.motif_rows, self.motif_cols
            )

        return weight_dict

    # --------------------- lightning hooks ---------------------------------
    def training_step(self, batch, batch_idx):
        x, y = batch
        weight_dict = self._generate_weights(
            hard=bool(self.hparams.hard),  # type: ignore[attr-defined]
        )

        logits = functional_call(self.target, weight_dict, (x,))
        task_loss = F.cross_entropy(logits, y)

        loss = task_loss

        with torch.no_grad():
            acc = logits.argmax(dim=1).eq(y).float().mean()

        self.log_dict(
            {
                "train/loss": loss,
                "train/task_loss": task_loss,
                "train/entropy": mean_entropy,
                "train/acc": acc,
                "train/tau": self._tau(),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # Hard argmax weights — matches deployment behaviour.
        weight_dict: dict[str, torch.Tensor] = {}
        for L_idx, name in enumerate(self._target_param_names):
            logits = self._layer_logits(L_idx)
            nbh, nbw = self._layer_block_counts(L_idx)
            idx = logits[:, :nbh, :nbw].argmax(dim=0)  # [nbh, nbw]
            blocks = self.motifs[idx]  # type: ignore[index]
            weight_dict[name] = blocks_to_tensor2D(
                blocks,
                self.layer_shapes[L_idx],
                self.motif_rows,
                self.motif_cols,
            )

        logits = functional_call(self.target, weight_dict, (x,))
        loss = F.cross_entropy(logits, y)
        acc = logits.argmax(dim=1).eq(y).float().mean()

        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_epoch=True, prog_bar=True)
        if acc.item() > self.best_val_acc:
            self.best_val_acc = float(acc.item())
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.hypernet.parameters(),
            lr=self.hparams.learning_rate,  # type: ignore[attr-defined]
            weight_decay=self.hparams.weight_decay,  # type: ignore[attr-defined]
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=self.hparams.epochs,  # type: ignore[attr-defined]
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def configure_gradient_clipping(
        self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None
    ):
        clip = float(self.hparams.grad_clip)  # type: ignore[attr-defined]
        if clip and clip > 0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=clip,
                gradient_clip_algorithm="norm",
            )
