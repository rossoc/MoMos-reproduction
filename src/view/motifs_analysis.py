import torch

from src.view.weight_distribution import extract_blocks, extract_blocks_2d, load_model


def unique_motifs_per_layer(blocks: list[torch.Tensor]) -> dict[str, dict]:
    """Return the number of unique motifs and utilisation ratio for each layer.

    Args:
        blocks: list of (n_blocks, block_dim) tensors, one per layer.

    Returns:
        dict keyed by "Layer 1", "Layer 2", ... with values:
            {
                "unique": int,   # number of distinct motifs
                "total":  int,   # total number of blocks in that layer
                "ratio":  float, # unique / total
            }
    """
    result = {}
    for i, layer_blocks in enumerate(blocks):
        unique = layer_blocks.unique(dim=0)
        n_unique = int(unique.size(0))
        n_total = int(layer_blocks.size(0))
        result[f"Layer {i + 1}"] = {
            "unique": n_unique,
            "total": n_total,
            "ratio": n_unique / n_total if n_total > 0 else 0.0,
        }
    return result


def unique_motifs_from_checkpoint(checkpoint_path: str, block_size: int) -> dict[str, dict]:
    """Load a checkpoint and return unique-motif counts per layer (1-D blocks)."""
    model = load_model(checkpoint_path)
    blocks, _ = extract_blocks(model, block_size)
    return unique_motifs_per_layer(blocks)


def unique_motifs_from_checkpoint_2d(
    checkpoint_path: str, rows: int, cols: int
) -> dict[str, dict]:
    """Load a checkpoint and return unique-motif counts per layer (2-D blocks)."""
    model = load_model(checkpoint_path)
    blocks, _ = extract_blocks_2d(model, rows, cols)
    return unique_motifs_per_layer(blocks)
