"""Lightning module: train the Mamba hypernetwork to *propose* a fresh MLP.

At every training step the hypernet emits per-position logits over motifs,
which are turned into a soft (τ-tempered) convex combination of motifs —
``W = softmax(logits / τ) @ M`` — tiled into MLP weight tensors. The target
MLP is called on a task batch via ``torch.func.functional_call`` and CE
flows back through the weights → softmax → hypernet, end-to-end
differentiable.

The softmax temperature ``tau`` is annealed linearly from ``tau_start`` to
``tau_end`` over the configured number of epochs — lower τ peaks the mix
toward a single motif per position.

Macro-batched updates: ``hypernet_update_every`` task batches share one set
of hypernet-generated weights (held as detached autograd leaves); their CE
gradients accumulate on those leaves, then ``_flush_macro_batch`` re-runs
the soft motif mix with grad enabled and does an exact VJP back into the
hypernet. The soft mix is deterministic in the hypernet params, so the
recompute is bit-identical to the leaf.

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
        # --- Memory ---
        motif_chunk_size: int | None = None,
        mamba_chunk_size: int | None = None,
        # --- Macro-batched hypernet updates ---
        hypernet_update_every: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["motifs", "layer_shapes"])
        # Manual optimization: we drive optimizer.step() ourselves so a single
        # chunked hypernet backward can amortize many cheap target-MLP steps.
        self.automatic_optimization = False

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

        self.motif_chunk_size = motif_chunk_size
        self.hypernet_update_every = max(1, int(hypernet_update_every))
        self._macro_step = 0
        self._W_leaves: dict[str, torch.Tensor] = {}
        self._val_weight_dict: dict[str, torch.Tensor] | None = None

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

        When grad is enabled and ``motif_chunk_size < K``, the K motifs are split
        into chunks and each chunk's forward is wrapped in
        ``torch.utils.checkpoint`` so peak activation memory stays bounded by
        one chunk.
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

    # --------------------- weight generation -------------------------------
    def _layer_block_counts(self, layer_idx: int) -> tuple[int, int]:
        shape = self.layer_shapes[layer_idx]
        orig_h, orig_w = int(shape[-2]), int(shape[-1])
        nbh = (orig_h + self.motif_rows - 1) // self.motif_rows
        nbw = (orig_w + self.motif_cols - 1) // self.motif_cols
        return nbh, nbw

    def _assemble_param(
        self, layer_idx: int, mix: torch.Tensor
    ) -> torch.Tensor:
        """Tile ``[K, nbh, nbw]`` motif weights into the layer's parameter shape."""
        blocks = torch.einsum("kd,khw->hwd", self.motifs, mix)  # [nbh, nbw, D]
        nbh, nbw = self._layer_block_counts(layer_idx)
        blocks = blocks.view(nbh * nbw, self.D)
        return blocks_to_tensor2D(
            blocks, self.layer_shapes[layer_idx], self.motif_rows, self.motif_cols
        )

    # --------------------- lightning hooks ---------------------------------
    def on_train_epoch_start(self):
        self._macro_step = 0
        self._W_leaves = {}

    def _soft_weights(self, layer_idx: int) -> torch.Tensor:
        """Soft (τ-tempered) motif mix for one layer: ``W = softmax(logits/τ) @ M``."""
        logits = self._layer_logits(layer_idx)                  # [K, nbh, nbw]
        soft = F.softmax(logits / self._tau(), dim=0)
        return self._assemble_param(layer_idx, soft)

    def _start_macro_batch(self) -> None:
        """Materialize fresh, detached weight leaves for a new macro-batch.

        Weights are a *soft* (τ-tempered) convex combination of motifs:
        ``W = softmax(logits/τ) @ M``. This is deterministic in the hypernet
        params, so the flush step can re-run the same computation with grad
        enabled and get bit-identical outputs — the VJP is exact.
        """
        self._W_leaves = {}
        with torch.no_grad():
            for L_idx, name in enumerate(self._target_param_names):
                W = self._soft_weights(L_idx)
                self._W_leaves[name] = W.detach().clone().requires_grad_(True)

    def _flush_macro_batch(self, optimizer) -> None:
        """Push accumulated upstream grads on ``W_leaves`` into the hypernet.

        Re-runs the soft motif mix with grad enabled — forward is bit-identical
        to the leaf the target MLP was forwarded on, so the VJP exactly equals
        the gradient at that operating point.
        """
        if not self._W_leaves:
            return

        names = list(self._W_leaves.keys())
        upstream = [self._W_leaves[n].grad for n in names]
        if any(grad is None for grad in upstream):
            # Macro started but no task batch contributed — bail.
            self._W_leaves = {}
            return

        outputs = [
            self._soft_weights(L_idx)
            for L_idx, name in enumerate(self._target_param_names)
            if name in self._W_leaves
        ]
        params = [p for p in self.hypernet.parameters() if p.requires_grad]
        grads = torch.autograd.grad(
            outputs=outputs,
            inputs=params,
            grad_outputs=upstream,
            allow_unused=True,
        )

        for p, gg in zip(params, grads):
            if gg is None:
                continue
            if p.grad is None:
                p.grad = gg.detach().clone()
            else:
                p.grad.add_(gg)

        clip = float(self.hparams.grad_clip)  # type: ignore[attr-defined]
        if clip and clip > 0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=clip,
                gradient_clip_algorithm="norm",
            )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        self._W_leaves = {}

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()

        if self._macro_step == 0:
            self._start_macro_batch()

        x, y = batch
        logits = functional_call(self.target, self._W_leaves, (x,))
        task_loss = F.cross_entropy(logits, y)
        # Only walks the cheap target-MLP graph; accumulates into W_leaves.grad.
        self.manual_backward(task_loss)

        with torch.no_grad():
            acc = logits.argmax(dim=1).eq(y).float().mean()

        self._macro_step += 1
        last_in_epoch = (batch_idx + 1) == self.trainer.num_training_batches
        if self._macro_step >= self.hypernet_update_every or last_in_epoch:
            self._flush_macro_batch(opt)
            self._macro_step = 0

        self.log_dict(
            {
                "train/loss": task_loss,
                "train/acc": acc,
                "train/tau": self._tau(),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return task_loss

    def on_train_epoch_end(self):
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()

    def on_validation_epoch_start(self):
        """Materialize the target MLP weight_dict once per validation epoch.

        Argmax weights are a deterministic function of the (frozen-during-val)
        hypernet params, so per-batch recomputation is pure waste.
        """
        with torch.no_grad():
            self._val_weight_dict = {
                name: self._soft_weights(L_idx)
                for L_idx, name in enumerate(self._target_param_names)
            }

    def on_validation_epoch_end(self):
        self._val_weight_dict = None

    def validation_step(self, batch, batch_idx):
        x, y = batch
        weight_dict = self._val_weight_dict
        assert weight_dict is not None, "on_validation_epoch_start did not run"

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
