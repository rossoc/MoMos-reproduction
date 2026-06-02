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
        out_channels,
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
        motif_batch_size: int | None = None,
        original_params: list | None = None,
        output_layers: int | None = None,
        loss_mode: str = "bce",
        ce_chunk_size: int | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["motifs", "layer_shapes", "image_datamodule", "original_params"]
        )
        if loss_mode not in ("bce", "chunked_ce"):
            raise ValueError(f"Unknown loss_mode {loss_mode!r}.")
        self.loss_mode = loss_mode
        self.ce_chunk_size = ce_chunk_size
        if loss_mode == "chunked_ce":
            self.automatic_optimization = False

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
            out_channels=out_channels,
            output_layers=output_layers,
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
        self.motif_batch_size = motif_batch_size
        self._cached_val_loader = None
        self.original_params = original_params
        self.best_val_acc = 0.0

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
        if self.loss_mode == "chunked_ce":
            return self._training_step_chunked_ce(batch, batch_idx)

        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y.float())

        pred = logits > 0
        acc = pred.eq(y.bool()).float().mean()

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def _forward_chunk_layer_logits(self, motif_ids, layer_id, n_rows, n_cols):
        """Run the Mamba model for a motif chunk and reshape its output so the
        flattened spatial dimension matches the layer grid layout ``[H_pad, W_pad]``.

        Returns: ``[c, N]`` where ``N = (n_rows*rows) * (n_cols*cols)``.
        """
        out = self.model(motif_ids, layer_id, int(n_rows), int(n_cols))
        c = out.shape[0]
        rows = int(self.rows)  # type: ignore
        cols = int(self.cols)  # type: ignore
        # out: [c, n_rows*n_cols, rows*cols] -> [c, n_rows, n_cols, rows, cols]
        out = out.view(c, int(n_rows), int(n_cols), rows, cols)
        # match grid layout: [c, n_rows, rows, n_cols, cols] -> [c, H_pad, W_pad]
        H_pad = int(n_rows) * rows
        W_pad = int(n_cols) * cols
        out = out.permute(0, 1, 3, 2, 4).reshape(c, H_pad * W_pad)
        return out

    def _training_step_chunked_ce(self, batch, batch_idx):
        """Two-pass chunked cross-entropy over motifs at each spatial position.

        Pass 1 (no grad): accumulate ``logZ[n] = logsumexp_k logits[k, n]`` chunk
        by chunk via ``logaddexp``. Pass 2 (with grad): for each chunk, build a
        surrogate loss whose gradient w.r.t. ``logits[k, n]`` equals the true CE
        gradient ``softmax(k, n) - 1[k=k_true]``, and call ``manual_backward``
        per chunk so only one chunk's activations are alive at a time.
        """
        (layer_id, n_rows, n_cols), grid = batch
        if isinstance(layer_id, torch.Tensor):
            layer_id_scalar = int(layer_id.item())
        else:
            layer_id_scalar = int(layer_id)
        n_rows = int(n_rows.item()) if isinstance(n_rows, torch.Tensor) else int(n_rows)
        n_cols = int(n_cols.item()) if isinstance(n_cols, torch.Tensor) else int(n_cols)
        device = self.device

        if self.motifs is None:
            raise RuntimeError("chunked_ce requires `motifs` to be registered.")
        K = int(self.motifs.shape[0])  # type: ignore
        chunk_size = int(self.ce_chunk_size) if self.ce_chunk_size else K
        chunk_size = max(1, min(chunk_size, K))

        grid_flat = grid.reshape(-1).to(device=device, dtype=torch.long)  # [N]
        valid = grid_flat >= 0
        n_valid = valid.sum().clamp(min=1).float()

        all_ids = torch.arange(K, device=device, dtype=torch.long)
        layer_id_t = torch.tensor(layer_id_scalar, device=device, dtype=torch.long)

        # ---- Pass 1: streaming logsumexp, no graph ----
        # NOTE: stay in train() mode so pass-2 logits match pass-1's logZ
        # exactly (the gradient identity relies on this).
        logZ = None
        with torch.no_grad():
            for s in range(0, K, chunk_size):
                ids = all_ids[s : s + chunk_size]
                chunk_logits = self._forward_chunk_layer_logits(
                    ids, layer_id_t, n_rows, n_cols
                )  # [c, N]
                chunk_lse = torch.logsumexp(chunk_logits.float(), dim=0)
                logZ = chunk_lse if logZ is None else torch.logaddexp(logZ, chunk_lse)
        assert logZ is not None

        # ---- Pass 2: chunked surrogate loss + manual_backward per chunk ----
        opt = self.optimizers()
        opt.zero_grad()

        total_loss_val = torch.zeros((), device=device)
        total_correct = torch.zeros((), device=device, dtype=torch.long)
        # Running argmax tracker for train accuracy
        best_val = torch.full((grid_flat.numel(),), float("-inf"), device=device)
        best_idx = torch.zeros(grid_flat.numel(), device=device, dtype=torch.long)

        valid_f = valid.float()

        for s in range(0, K, chunk_size):
            ids = all_ids[s : s + chunk_size]
            chunk_logits = self._forward_chunk_layer_logits(
                ids, layer_id_t, n_rows, n_cols
            )  # [c, N], requires grad
            chunk_f = chunk_logits.float()

            with torch.no_grad():
                w = (chunk_f - logZ.unsqueeze(0)).exp() * valid_f.unsqueeze(0)  # softmax slice

                # Update train-acc tracker
                cmax, cargmax = chunk_f.max(dim=0)
                update = cmax > best_val
                best_val = torch.where(update, cmax, best_val)
                best_idx = torch.where(update, ids[cargmax], best_idx)

            # Surrogate term whose gradient is softmax(k,n) - 1[k=k_true]
            in_chunk = (grid_flat.unsqueeze(0) == ids.unsqueeze(1)) & valid.unsqueeze(0)
            neg = (w * chunk_f).sum()
            pos = (chunk_f * in_chunk.float()).sum()
            loss_chunk = (neg - pos) / n_valid

            self.manual_backward(loss_chunk)

            with torch.no_grad():
                # True per-chunk loss-value contribution (logZ - logit_true on positions in chunk)
                pos_val = (chunk_f.detach() * in_chunk.float()).sum()
                logZ_val = (logZ * in_chunk.any(dim=0).float()).sum()
                total_loss_val = total_loss_val + (logZ_val - pos_val)

        opt.step()

        with torch.no_grad():
            total_correct = ((best_idx == grid_flat) & valid).sum()
            train_loss = total_loss_val / n_valid
            train_acc = total_correct.float() / n_valid

        self.log("train/loss", train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return None

    def build_mlp(
        self, motifs: torch.Tensor | None = None
    ) -> tuple[LitMLP, float | None]:
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

        sse_total = 0.0
        n_total = 0
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                params = list(iter_trainable_params(lit_mlp.model))
                for L_idx, param in enumerate(params):
                    layer_h = int(self.n_rows_per_layer[L_idx])  # type: ignore
                    layer_w = int(self.n_cols_per_layer[L_idx])  # type: ignore
                    H_pad = layer_h * int(self.rows)  # type: ignore
                    W_pad = layer_w * int(self.cols)  # type: ignore

                    logits_full = torch.empty(K, H_pad, W_pad, device=device)
                    layer_id = torch.tensor(L_idx, device=device, dtype=torch.long)
                    all_ids = torch.arange(K, device=device, dtype=torch.long)
                    chunk = int(self.motif_batch_size) if self.motif_batch_size else K
                    for start in range(0, K, chunk):
                        ids = all_ids[start : start + chunk]
                        out = self.model(ids, layer_id, layer_h, layer_w)
                        b = out.shape[0]
                        out = out.view(
                            b,
                            layer_h,
                            layer_w,
                            int(self.rows),
                            int(self.cols),
                        )
                        out = out.permute(0, 1, 3, 2, 4).reshape(b, H_pad, W_pad)
                        logits_full[start : start + chunk] = out

                    predicted_M_padded = logits_full.argmax(dim=0)

                    shape = self.layer_shapes[L_idx]
                    orig_h, orig_w = int(shape[-2]), int(shape[-1])
                    num_blocks_h = (orig_h + int(self.motif_rows) - 1) // int(
                        self.motif_rows
                    )
                    num_blocks_w = (orig_w + int(self.motif_cols) - 1) // int(
                        self.motif_cols
                    )
                    predicted_M = predicted_M_padded[:num_blocks_h, :num_blocks_w]

                    blocks = motifs_t[predicted_M]
                    blocks = blocks.reshape(
                        num_blocks_h * num_blocks_w,
                        int(self.motif_rows) * int(self.motif_cols),
                    )
                    reconstructed = blocks_to_tensor2D(
                        blocks, shape, int(self.motif_rows), int(self.motif_cols)
                    )
                    recon = reconstructed.view_as(param).to(param.dtype)
                    param.data.copy_(recon)

                    if self.original_params is not None:
                        orig = self.original_params[L_idx].to(
                            device=device, dtype=recon.dtype
                        )
                        sse_total += float(((recon - orig) ** 2).sum().item())
                        n_total += orig.numel()
        finally:
            if was_training:
                self.train()

        mse = (sse_total / n_total) if n_total > 0 else None
        return lit_mlp, mse

    def on_train_epoch_end(self):
        """Reconstruct an MLP from predicted masks and evaluate on the validation fold."""
        # Step the LR scheduler manually when running with manual optimization.
        if not self.automatic_optimization:
            sch = self.lr_schedulers()
            if sch is not None:
                sch.step()

        if self.image_datamodule is None or self.motifs is None:
            return

        if self._cached_val_loader is None:
            self._cached_val_loader = self.image_datamodule.val_dataloader()
        val_loader = self._cached_val_loader
        if val_loader is None:
            return

        lit_mlp, recon_mse = self.build_mlp()
        if recon_mse is not None:
            self.log("val/recon_mse", recon_mse, prog_bar=True)
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

        val_acc = total_correct / n
        self.log("val/loss", total_loss / n, prog_bar=True)
        self.log("val/acc", val_acc, prog_bar=True)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc

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
