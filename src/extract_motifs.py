"""Extract motif dictionary and per-block motif assignments from a MoMos-quantized checkpoint.

After MoMos2D quantization, every weight block equals one of K unique motif
vectors. This script:
  1. Loads a checkpoint (Lightning- or raw-state-dict).
  2. Concatenates per-tensor blocks via ``tensor2D_to_blocks`` in state-dict
     iteration order (same order used by ``BlocksDataModule``).
  3. Finds the unique blocks (the motifs) and maps each block index to its
     motif id via ``torch.unique(..., return_inverse=True)``.
  4. Saves ``{motifs, assignments, rows, cols, num_blocks, num_motifs}`` to disk.

Usage:
    python src/extract_motifs.py --ckpt final.pt --rows 1 --cols 2 --out motifs.pt
"""

import argparse
import os

import torch
import torch.nn as nn

from quantizers.momos2d import tensor2D_to_blocks
from utils.io_model import load_model_from_checkpoint


def _extract_state_dict(loaded) -> dict:
    sd = loaded["state_dict"]
    if isinstance(sd, nn.Module):
        sd = sd.state_dict()
    # Strip a leading "model." prefix if present (Lightning wrapping).
    if any(k.startswith("model.") for k in sd):
        sd = {k[len("model.") :]: v for k, v in sd.items() if k.startswith("model.")}
    return sd


def extract_motifs(state_dict: dict, rows: int, cols: int):
    """Return (motifs, assignments, layer_specs).

    motifs: (K, rows*cols) float tensor of unique blocks.
    assignments: (N,) long tensor; assignments[i] = motif id of block i.
    layer_specs: list of (name, n_blocks, original_shape) preserving the block order.
    """
    all_blocks: list[torch.Tensor] = []
    layer_specs: list[tuple[str, int, torch.Size]] = []

    for name, t in state_dict.items():
        if not torch.is_tensor(t):
            continue
        print(t.shape)
        blocks, _, shape = tensor2D_to_blocks(t.detach().float(), rows, cols)
        layer_specs.append((name, int(blocks.size(0)), shape))
        all_blocks.append(blocks)

    if not all_blocks:
        raise RuntimeError("No floating-point tensors found in state_dict.")

    y = torch.cat(all_blocks, dim=0)  # (N, rows*cols)
    motifs, assignments = torch.unique(y, dim=0, return_inverse=True)
    return motifs, assignments.long(), layer_specs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to MoMos-quantized checkpoint.")
    p.add_argument("--rows", type=int, required=True)
    p.add_argument("--cols", type=int, required=True)
    p.add_argument("--out", default="motifs.pt", help="Output .pt file.")
    args = p.parse_args()

    loaded = load_model_from_checkpoint(args.ckpt)
    if loaded is None:
        raise FileNotFoundError(args.ckpt)
    sd = _extract_state_dict(loaded)

    motifs, assignments, layer_specs = extract_motifs(sd, args.rows, args.cols)

    n_blocks = int(assignments.numel())
    n_motifs = int(motifs.size(0))

    # Per-block layer id (0..L-1) and within-layer index (resets at each tensor).
    layer_ids = torch.empty(n_blocks, dtype=torch.long)
    within_layer_idx = torch.empty(n_blocks, dtype=torch.long)
    offset = 0
    for layer_id, (_, n_b, _) in enumerate(layer_specs):
        layer_ids[offset : offset + n_b] = layer_id
        within_layer_idx[offset : offset + n_b] = torch.arange(n_b)
        offset += n_b
    num_layers = len(layer_specs)
    counts = torch.bincount(assignments, minlength=n_motifs)

    print(f"Loaded checkpoint:    {args.ckpt}")
    print(f"Tensors processed:    {len(layer_specs)}")
    print(
        f"Block size:           rows={args.rows}, cols={args.cols}  (dim={args.rows * args.cols})"
    )
    print(f"Total blocks (N):     {n_blocks}")
    print(f"Unique motifs (K):    {n_motifs}")
    print(f"Compression ratio:    {n_blocks / max(1, n_motifs):.1f}x")
    print(
        f"Largest motif covers: {int(counts.max())} blocks ({100 * counts.max() / n_blocks:.1f}%)"
    )
    print(f"Smallest motif covers:{int(counts.min())} blocks")

    out = {
        "motifs": motifs,  # (K, rows*cols)
        "assignments": assignments,  # (N,) long, indexed 0..K-1
        "layer_ids": layer_ids,  # (N,) long, in [0, L)
        "within_layer_idx": within_layer_idx,  # (N,) long, resets at each tensor
        "num_layers": num_layers,
        "rows": int(args.rows),
        "cols": int(args.cols),
        "num_blocks": n_blocks,
        "num_motifs": n_motifs,
        "layer_specs": [(n, nb, tuple(s)) for n, nb, s in layer_specs],
        "source_checkpoint": os.path.abspath(args.ckpt),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    torch.save(out, args.out)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
