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


def build_swap_motif(from_percentiles, to_percentiles, probability, window=0.01):
    def swap(idx):
        motifs_idx, counts = idx.unique(return_counts=True)
        # Sort so we can work in "percentile space"
        sorted_indices = torch.argsort(counts)
        motifs_sorted = motifs_idx[sorted_indices]
        num_unique = len(motifs_sorted)

        new_idx = idx.clone()
        prob_mask = torch.rand(idx.shape, device=idx.device) < probability

        for f_perc, t_perc in zip(from_percentiles, to_percentiles):
            # 1. Define the Rank Ranges
            # e.g., if f_perc is 0.5 and window is 0.01, we look at 0.49 to 0.51
            # Replace your range definitions with this:
            f_low = int(round(max(0, f_perc - window) * (num_unique - 1)))
            f_high = int(round(min(1.0, f_perc + window) * (num_unique - 1)))

            t_low = int(round(max(0, t_perc - window) * (num_unique - 1)))
            t_high = int(round(min(1.0, t_perc + window) * (num_unique - 1)))

            # Then slice using:
            source_motifs = motifs_sorted[f_low : f_high + 1]
            target_motifs = motifs_sorted[t_low : t_high + 1]

            if len(source_motifs) == 0 or len(target_motifs) == 0:
                print("Error!!!")
                continue

            # 3. Apply Swaps
            # For each source motif in the range, pick a target motif to replace it with
            for s_motif in source_motifs:
                # Pick a random motif from the target window to swap into
                # (You could also just pick the median of the target window)
                t_idx = torch.randint(0, len(target_motifs), (1,)).item()
                t_motif = target_motifs[t_idx]

                swap_mask = (idx == s_motif) & prob_mask
                new_idx[swap_mask] = t_motif

        return new_idx

    return swap


# %%
if __name__ == "__main__":
    torch.manual_seed(0)

    # 100 indices drawn from 4 motif IDs with skewed frequencies:
    # motif 0 appears ~50x (high freq), motif 3 appears ~5x (low freq)
    idx = torch.cat(
        [
            torch.zeros(
                50, dtype=torch.long
            ),  # motif 0: most frequent  (top percentile)
            torch.ones(25, dtype=torch.long),  # motif 1
            torch.full((20,), 2, dtype=torch.long),  # motif 2
            torch.full(
                (5,), 3, dtype=torch.long
            ),  # motif 3: least frequent (bottom percentile)
        ]
    )
    idx = idx[torch.randperm(len(idx))]  # shuffle

    # Swap low-frequency motifs (bottom ~10%) → high-frequency motifs (top ~10%)
    swap = build_swap_motif(
        from_percentiles=[0.0],  # bottom of rank distribution (least frequent)
        to_percentiles=[1.0],  # top of rank distribution (most frequent)
        probability=1.0,  # always swap when mask hits
        window=0.1,
    )

    new_idx = swap(idx)

    original_counts = {i: (idx == i).sum().item() for i in range(4)}
    new_counts = {i: (new_idx == i).sum().item() for i in range(4)}

    print("Original counts:", original_counts)
    print("New counts:     ", new_counts)
    # Expect: motif 3 (least frequent) count drops toward 0,
    #         motif 0 (most frequent) count increases.
    assert new_counts[3] < original_counts[3], "low-freq motif should be swapped out"
    assert new_counts[0] > original_counts[0], "high-freq motif should gain indices"
    print("Test passed.")


def _nearest_motifs_chunked(
    blocks,
    motifs,
    chunk_size=None,
    show_progress=False,
    progress_prefix="momos",
    progress_every_elements=None,
):
    """Assign each block to nearest motif using chunked pairwise distances.

    Args:
        blocks: Block matrix with shape ``[num_blocks, block_size]``.
        motifs: Motif matrix with shape ``[k_eff, block_size]``.
        chunk_size: Optional memory budget in MB for chunked distance
            computation. Default is ``4096`` MB (~4 GB).
        show_progress: If True, print coarse chunk progress updates.
        progress_prefix: Label prefix for progress lines.
        progress_every_elements: Optional progress print interval measured in
            processed scalar elements.

    Returns:
        Long tensor of nearest motif indices for each block.
    """
    n_blocks = int(blocks.size(0))
    block_size = int(blocks.size(1)) if blocks.dim() > 1 else 1
    n_motifs = int(motifs.size(0))
    chunk_size = chunk_size or 256
    chunk = _resolve_chunk_size_blocks(
        n_blocks,
        n_motifs,
        chunk_size=chunk_size,
        dtype=blocks.dtype,
    )

    nearest = torch.empty(n_blocks, dtype=torch.long, device=blocks.device)
    motifs_t = motifs.t().contiguous()
    motifs_norm2 = motifs.square().sum(dim=1).view(1, -1)
    total_elements = n_blocks * block_size
    total_chunks = (n_blocks + chunk - 1) // chunk
    print_every = _resolve_progress_every_elements(
        total_elements,
        progress_every_elements=progress_every_elements,
    )
    next_emit = print_every
    for start in range(0, n_blocks, chunk):
        end = min(start + chunk, n_blocks)
        chunk_blocks = blocks[start:end]
        chunk_norm2 = chunk_blocks.square().sum(dim=1, keepdim=True)
        # Exact Euclidean argmin via squared distances:
        # ||x-c||^2 = ||x||^2 + ||c||^2 - 2 x c^T
        nearest[start:end] = torch.addmm(
            chunk_norm2 + motifs_norm2, chunk_blocks, motifs_t, beta=1, alpha=-2.0
        ).argmin(dim=1)

        if show_progress:
            chunk_idx = (start // chunk) + 1
            local_done = end * block_size
            should_emit = local_done >= next_emit or end == n_blocks
            if should_emit:
                pct = 100.0 * float(local_done) / float(max(1, total_elements))
                blocks_done = int(end)
                blocks_total = int(n_blocks)
                pairwise_done = int(blocks_done * n_motifs)
                pairwise_total = int(blocks_total * n_motifs)
                print(
                    f"{progress_prefix}: chunk {chunk_idx}/{total_chunks} "
                    f"blocks {blocks_done:,}/{blocks_total:,} "
                    f"pairwise {pairwise_done:,}/{pairwise_total:,} ({pct:.1f}%)",
                    flush=True,
                )
                while next_emit <= local_done:
                    next_emit += print_every
    return nearest

def _initialize_motifs(all_blocks, k_eff, block_size, force_zero):
    """Handles motif selection based on force_zero logic."""
    total_blocks = all_blocks.size(0)
    motifs = torch.zeros(
        k_eff, block_size, device=all_blocks.device, dtype=all_blocks.dtype
    )

    if force_zero and k_eff > 1:
        idx = torch.randperm(total_blocks, device=all_blocks.device)[: k_eff - 1]
        motifs[1:] = all_blocks[idx]  # First row remains zero
    elif not force_zero:
        idx = torch.randperm(total_blocks, device=all_blocks.device)[:k_eff]
        motifs = all_blocks[idx]

    return motifs

def _assign_blocks(
    all_blocks, motifs, chunk_size, show_progress, prefix, progress_every, swapping_fn
):
    """Finds nearest motifs and applies optional swapping."""
    total_blocks = all_blocks.size(0)

    # Handle the edge case where only one motif exists (usually force_zero=True, k=1)
    if motifs.size(0) == 1:
        nearest = torch.zeros(total_blocks, dtype=torch.long, device=all_blocks.device)
        return nearest, 0

    nearest = _nearest_motifs_chunked(
        all_blocks,
        motifs,
        chunk_size=chunk_size,
        show_progress=show_progress,
        progress_prefix=prefix,
        progress_every_elements=progress_every,
    )

    swapped_count = 0
    if swapping_fn is not None:
        swapped = swapping_fn(nearest)
        swapped_count = (nearest != swapped).sum().item()
        nearest = swapped

    return nearest, swapped_count
