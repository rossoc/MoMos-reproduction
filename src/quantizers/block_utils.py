"""Block conversion utilities and chunking helpers for MoMos quantization."""

import torch


def iter_trainable_params(model):
    """Yield non-empty trainable parameters from a model.

    Args:
        model: Model to iterate.

    Yields:
        Parameter tensors with ``requires_grad=True``.
    """
    for param in model.parameters():
        if param.requires_grad and param.numel() > 0:
            yield param


def tensor_to_blocks(tensor, block_size):
    """Flatten and pad a tensor into fixed-size blocks.

    Args:
        tensor: Source parameter tensor.
        block_size: Number of values per block.

    Returns:
        Tuple ``(blocks, original_numel, original_shape)``.
    """
    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}")
    flat = tensor.flatten()
    n_params = flat.numel()
    pad = (-n_params) % block_size
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, device=flat.device, dtype=flat.dtype)])
    return flat.view(-1, block_size), n_params, tensor.shape


def blocks_to_tensor(blocks, n_params, shape):
    """Reconstruct original tensor shape from flattened blocks.

    Args:
        blocks: Block matrix.
        n_params: Number of original (unpadded) values.
        shape: Original tensor shape.

    Returns:
        Tensor restored to original shape.
    """
    return blocks.flatten()[:n_params].view(shape)


def count_total_blocks(model, block_size):
    """Count quantization blocks across all trainable parameters.

    Args:
        model: Model to inspect.
        block_size: Number of values per block.

    Returns:
        Integer number of blocks.
    """
    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}")
    total = 0
    for param in iter_trainable_params(model):
        total += (param.numel() + block_size - 1) // block_size
    return int(total)


def k_from_capacity(model, block_size, capacity):
    """Convert capacity ratio to absolute motif count ``k``.

    Args:
        model: Model used to determine total block count.
        block_size: Number of values per block.
        capacity: Fraction of blocks to keep as motifs.

    Returns:
        Integer ``k`` clamped to ``[1, total_blocks]``.
    """
    capacity = float(capacity)
    if capacity <= 0:
        raise ValueError("capacity must be > 0")
    n_blocks = count_total_blocks(model, block_size)
    k = int(capacity * n_blocks)
    return max(1, min(k, n_blocks))


def _resolve_chunk_size_blocks(
    n_blocks, n_motifs, chunk_size=4096, dtype=torch.float32
):
    """Resolve chunk size (in blocks) for block-vs-motif distance computation.

    Args:
        n_blocks: Number of blocks being assigned.
        n_motifs: Number of motifs compared against.
        chunk_size: Optional memory budget in MB for the distance matrix.
            Default is ``4096`` MB (~4 GB) when ``None``.
        dtype: Floating dtype used in distance matrix.

    Returns:
        Integer chunk size in ``[1, n_blocks]``.
    """
    n_blocks = int(n_blocks)
    n_motifs = max(1, int(n_motifs))
    if n_blocks <= 0:
        return 1

    chunk_size_mb = float(chunk_size)
    if chunk_size_mb <= 0:
        raise ValueError(f"chunk_size must be > 0 MB, got {chunk_size}")
    bytes_budget = int(chunk_size_mb * 1024 * 1024)
    bytes_per = int(torch.empty((), dtype=dtype).element_size())
    max_elems = max(1, bytes_budget // max(1, bytes_per))
    chunk = max(1, max_elems // n_motifs)
    return min(n_blocks, chunk)


def _resolve_progress_every_elements(
    total_elements, progress_every_elements=None, n_reports=20
):
    """Resolve how often progress should be printed by element count.

    Args:
        total_elements: Total scalar elements processed in one call.
        progress_every_elements: Optional explicit reporting interval.

    Returns:
        Positive integer reporting interval in elements.
    """
    total_elements = int(total_elements)
    if total_elements <= 0:
        return 1
    if progress_every_elements is not None:
        value = int(progress_every_elements)
        if value <= 0:
            raise ValueError(
                f"progress_every_elements must be > 0, got {progress_every_elements}"
            )
        return value
    # Default: about 20 progress updates per call.
    return max(1, total_elements // n_reports)
