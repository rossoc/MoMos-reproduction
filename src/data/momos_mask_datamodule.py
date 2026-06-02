"""DataModule that turns a momos2d-quantized MLP checkpoint into
per-(motif, layer) binary-mask training samples for the Mamba model."""

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import lightning as L

from view.weight_distribution import load_model, extract_blocks_2d
from quantizers.momos import _nearest_motifs_chunked
from quantizers.momos2d import pad_to_multiple


class MotifMaskDataset(Dataset):
    """Yields training samples for the Mamba mask-predictor.

    Two modes:
      - ``per_motif`` (default): one item == one (motif, layer) pair, target is
        the binary mask ``(grid == motif_idx)`` partitioned into ``rows x cols``
        subblocks. Used with ``LayerBucketBatchSampler`` for BCE training.
      - ``per_layer``: one item == one layer, target is the full motif-index
        grid ``[H_pad, W_pad]`` (with ``-1`` for padded positions). Used with
        chunked cross-entropy training in ``LitMamba``.
    """

    def __init__(
        self,
        checkpoint_path: str,
        rows: int,
        cols: int,
        motif_rows: int,
        motif_cols: int,
        chunk_size=256,
        mode: str = "per_motif",
    ):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.rows = int(rows)
        self.cols = int(cols)
        if mode not in ("per_motif", "per_layer"):
            raise ValueError(f"Unknown mode {mode!r}; expected per_motif or per_layer.")
        self.mode = mode

        model = load_model(checkpoint_path)
        if motif_rows is not None and motif_cols is not None:
            self.motif_rows, self.motif_cols = int(motif_rows), int(motif_cols)
        else:
            raise ValueError(
                "Instatiating MtoifMaskDataset, but no motif_row or motif_cols provided"
            )

        blocks, layers_specs = extract_blocks_2d(
            model, self.motif_rows, self.motif_cols
        )
        all_blocks = torch.cat(blocks, dim=0)
        self.motifs = all_blocks.unique(dim=0)
        M = _nearest_motifs_chunked(all_blocks, self.motifs, chunk_size)

        self.layer_grids = []
        self.layer_shapes = [spec[3] for spec in layers_specs]
        self.original_params = [spec[0].detach().clone() for spec in layers_specs]
        # Per-layer original blocks: [num_blocks, motif_rows, motif_cols] in the
        # same flat order as `grid.view(layer_h, layer_w)` (row-major).
        self.layer_blocks = [b.detach().clone() for b in blocks]
        offset = 0
        for _, num_blocks, _, shape in layers_specs:
            layer_h = shape[0] // self.motif_rows
            layer_w = shape[1] // self.motif_cols
            assert layer_h * layer_w == num_blocks, (
                f"Layer block-count mismatch: {layer_h}*{layer_w} != {num_blocks}"
            )
            grid = M[offset : offset + num_blocks].view(layer_h, layer_w)
            offset += num_blocks

            grid = pad_to_multiple(grid, self.rows, self.cols, value=-1)

            self.layer_grids.append(grid)

        self.n_motifs = int(self.motifs.shape[0])
        self.n_layers = len(self.layer_grids)
        self.n_rows_per_layer = [g.shape[0] // self.rows for g in self.layer_grids]
        self.n_cols_per_layer = [g.shape[1] // self.cols for g in self.layer_grids]

    def __len__(self):
        if self.mode == "per_layer":
            return self.n_layers
        return self.n_motifs * self.n_layers

    def __getitem__(self, index):
        if self.mode == "per_layer":
            layer_idx = index
            grid = self.layer_grids[layer_idx]
            inputs = (
                torch.tensor(layer_idx, dtype=torch.long),
                int(self.n_rows_per_layer[layer_idx]),
                int(self.n_cols_per_layer[layer_idx]),
            )
            return inputs, grid.long()

        motif_idx = index // self.n_layers
        layer_idx = index % self.n_layers

        grid = self.layer_grids[layer_idx]
        n_rows = self.n_rows_per_layer[layer_idx]
        n_cols = self.n_cols_per_layer[layer_idx]

        mask = (grid == motif_idx).float()
        blocks2d = mask.unfold(0, self.rows, self.rows).unfold(1, self.cols, self.cols)
        target = blocks2d.reshape(n_rows * n_cols, self.rows * self.cols).contiguous()

        inputs = (
            torch.tensor(motif_idx, dtype=torch.long),
            torch.tensor(layer_idx, dtype=torch.long),
            int(n_rows),
            int(n_cols),
        )
        return inputs, target


class LayerBucketBatchSampler(Sampler[list[int]]):
    """Yields batches of dataset indices that all share the same layer_idx.

    MotifMaskDataset uses `motif_idx = index // n_layers`,
    `layer_idx = index % n_layers`, so indices with the same
    `index % n_layers` belong to the same layer.
    """

    def __init__(
        self,
        n_motifs: int,
        n_layers: int,
        batch_size: int,
        shuffle: bool = True,
        generator: torch.Generator | None = None,
    ):
        self.n_motifs = int(n_motifs)
        self.n_layers = int(n_layers)
        self.batch_size = max(1, int(batch_size))
        self.shuffle = bool(shuffle)
        self.generator = generator

    def __iter__(self):
        perm_kwargs = {"generator": self.generator} if self.generator is not None else {}
        buckets: list[list[int]] = []
        for L_idx in range(self.n_layers):
            idx = torch.arange(self.n_motifs, dtype=torch.long) * self.n_layers + L_idx
            if self.shuffle:
                idx = idx[torch.randperm(idx.numel(), **perm_kwargs)]
            for s in range(0, idx.numel(), self.batch_size):
                buckets.append(idx[s : s + self.batch_size].tolist())
        if self.shuffle:
            order = torch.randperm(len(buckets), **perm_kwargs).tolist()
            buckets = [buckets[i] for i in order]
        yield from buckets

    def __len__(self):
        per_layer = (self.n_motifs + self.batch_size - 1) // self.batch_size
        return per_layer * self.n_layers


def _layer_collate(samples):
    inputs0 = samples[0][0]
    layer_id = inputs0[1]
    n_rows, n_cols = inputs0[2], inputs0[3]
    motif_ids = torch.stack([s[0][0] for s in samples], dim=0)
    targets = torch.stack([s[1] for s in samples], dim=0)
    return (motif_ids, layer_id, n_rows, n_cols), targets


def _per_layer_collate(samples):
    """Collate one per_layer sample (batch_size must be 1) without adding a batch dim."""
    assert len(samples) == 1, "per_layer mode requires batch_size=1"
    (layer_id, n_rows, n_cols), grid = samples[0]
    return (layer_id, n_rows, n_cols), grid


class MotifMaskDataModule(L.LightningDataModule):
    def __init__(
        self,
        checkpoint_path: str,
        rows: int,
        cols: int,
        motif_rows: int,
        motif_cols: int,
        batch_size: int = 1,
        runtime: dict | None = None,
        motif_batch_size: int | None = None,
        mode: str = "per_motif",
    ):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.rows = int(rows)
        self.cols = int(cols)
        self.motif_rows = motif_rows
        self.motif_cols = motif_cols
        self.batch_size = int(batch_size)
        self.motif_batch_size = motif_batch_size
        self.mode = mode
        self.runtime = runtime or {
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "prefetch_factor": None,
        }
        self.train_dataset: MotifMaskDataset | None = None

    def setup(self, stage=None):
        if self.train_dataset is not None:
            return
        self.train_dataset = MotifMaskDataset(
            self.checkpoint_path,
            self.rows,
            self.cols,
            motif_rows=self.motif_rows,
            motif_cols=self.motif_cols,
            mode=self.mode,
        )

    def _build_dataloader(self, dataset, shuffle=False) -> DataLoader:
        kwargs: dict = {
            "num_workers": int(self.runtime["num_workers"]),  # type: ignore
            "pin_memory": bool(self.runtime["pin_memory"]),
        }
        if kwargs["num_workers"] > 0:
            kwargs["persistent_workers"] = bool(self.runtime["persistent_workers"])
            if self.runtime["prefetch_factor"] is not None:
                kwargs["prefetch_factor"] = int(self.runtime["prefetch_factor"])

        if self.mode == "per_layer":
            kwargs["batch_size"] = 1
            kwargs["shuffle"] = shuffle
            kwargs["collate_fn"] = _per_layer_collate
            return DataLoader(dataset, **kwargs)  # type: ignore

        use_bucketed = self.motif_batch_size is not None and int(self.motif_batch_size) > 1
        if use_bucketed:
            sampler = LayerBucketBatchSampler(
                n_motifs=dataset.n_motifs,
                n_layers=dataset.n_layers,
                batch_size=int(self.motif_batch_size),
                shuffle=shuffle,
            )
            kwargs["batch_sampler"] = sampler
            kwargs["collate_fn"] = _layer_collate
        else:
            kwargs["batch_size"] = self.batch_size
            kwargs["shuffle"] = shuffle
            kwargs["collate_fn"] = lambda batch: batch[0]
        return DataLoader(dataset, **kwargs)  # type: ignore

    def train_dataloader(self):
        return self._build_dataloader(self.train_dataset, shuffle=True)

    def summary(self) -> dict:
        if self.train_dataset is None:
            self.setup()

        assert self.train_dataset

        ds = self.train_dataset
        num_rows = max(ds.n_rows_per_layer)
        num_cols = max(ds.n_cols_per_layer)
        return {
            "n_motifs": ds.n_motifs,
            "num_rows": num_rows,
            "num_cols": num_cols,
            "num_layers": ds.n_layers,
            "out_channels": self.rows * self.cols,
        }
