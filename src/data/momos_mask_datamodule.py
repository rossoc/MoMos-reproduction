"""DataModule that turns a momos2d-quantized MLP checkpoint into
per-(motif, layer) binary-mask training samples for the Mamba model."""

import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L

from view.weight_distribution import load_model, extract_blocks_2d
from quantizers.momos import _nearest_motifs_chunked
from quantizers.momos2d import pad_to_multiple


class MotifMaskDataset(Dataset):
    """Yields ((motif_idx, layer_idx, n_rows, n_cols), target_blocks).

    For every (motif, layer) pair, computes the binary mask
    `(M2d_layer == motif_idx)`, partitions it into `rows x cols` blocks,
    and flattens each block into the channel dimension of the target.
    """

    def __init__(
        self,
        checkpoint_path: str,
        rows: int,
        cols: int,
        motif_rows: int,
        motif_cols: int,
        chunk_size=256,
    ):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.rows = int(rows)
        self.cols = int(cols)

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
        return self.n_motifs * self.n_layers

    def __getitem__(self, index):
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
    ):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.rows = int(rows)
        self.cols = int(cols)
        self.motif_rows = motif_rows
        self.motif_cols = motif_cols
        self.batch_size = int(batch_size)
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
        )

    def _build_dataloader(self, dataset, shuffle=False) -> DataLoader:
        kwargs = {
            "batch_size": self.batch_size,
            "shuffle": shuffle,
            "num_workers": int(self.runtime["num_workers"]),  # type: ignore
            "pin_memory": bool(self.runtime["pin_memory"]),
            "collate_fn": lambda batch: batch[0],
        }
        if kwargs["num_workers"] > 0:  # type: ignore
            kwargs["persistent_workers"] = bool(self.runtime["persistent_workers"])
            if self.runtime["prefetch_factor"] is not None:
                kwargs["prefetch_factor"] = int(self.runtime["prefetch_factor"])
        return DataLoader(dataset, **kwargs)  # type: ignore

    def train_dataloader(self):
        return self._build_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return None

    def test_dataloader(self):
        return None

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
            "cnn_out_channels": self.rows * self.cols,
        }
