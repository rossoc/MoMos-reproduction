"""Core MoMos quantization algorithm, dispatcher, and MoMos class."""

import time

import torch

from .block_utils import (
    iter_trainable_params,
    tensor_to_blocks,
    blocks_to_tensor,
    k_from_capacity,
    _resolve_chunk_size_blocks,
    _resolve_progress_every_elements,
    build_swap_motif,
)
from .fake_quant import quantize_qat


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
    chunk_size = chunk_size or 4096
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


def momos(
    model,
    block_size,
    k,
    force_zero=True,
    chunk_size=None,
    show_chunk_progress=False,
    progress_prefix="momos",
    progress_every_elements=None,
    swapping_fn=None,
):
    """Apply one MoMos projection step.

    Args:
        model: Model to quantize in-place.
        block_size: Block size ``s``.
        k: Number of motifs.
        force_zero: If True, include all-zero motif.
        chunk_size: Optional memory budget in MB for nearest-motif distance
            chunking. ``None`` uses the default ``4096`` MB (~4 GB).
        show_chunk_progress: If True, print coarse chunk progress updates.
        progress_prefix: Label prefix for chunk progress lines.
        progress_every_elements: Optional progress print interval measured in
            processed scalar elements.

    Motifs are selected globally across all trainable blocks, then each
    block is assigned to its nearest motif (also globally).

    Returns:
        Dict with distortion, changed weights, and motif usage counts.
    """
    block_size = int(block_size)
    k = int(k)
    motif_counts = torch.zeros(max(1, k), dtype=torch.long)

    with torch.no_grad():
        layer_specs = []
        all_blocks = []

        for param in iter_trainable_params(model):
            blocks, n_params, shape = tensor_to_blocks(param.detach(), block_size)
            layer_specs.append((param, int(blocks.size(0)), int(n_params), shape))
            all_blocks.append(blocks)

        if not all_blocks:
            return {
                "distortion": 0.0,
                "num_changed_weights": 0,
                "motif_counts": motif_counts,
            }

        all_blocks = torch.cat(all_blocks, dim=0)
        total_blocks = int(all_blocks.size(0))
        k_eff = max(1, min(k, total_blocks))

        motifs = torch.zeros(
            k_eff, block_size, device=all_blocks.device, dtype=all_blocks.dtype
        )
        if force_zero and k_eff > 1:
            idx = torch.randint(0, total_blocks, (k_eff - 1,), device=all_blocks.device)
            motifs = all_blocks[idx]

        elif not force_zero:
            idx = torch.randperm(total_blocks, device=all_blocks.device)[:k_eff]
            motifs[1:] = all_blocks[idx]

        if force_zero and k_eff == 1:
            nearest = torch.zeros(
                total_blocks, dtype=torch.long, device=all_blocks.device
            )
            quantized_blocks = motifs.expand(total_blocks, -1)
        else:
            nearest = _nearest_motifs_chunked(
                all_blocks,
                motifs,
                chunk_size=chunk_size,
                show_progress=bool(show_chunk_progress),
                progress_prefix=str(progress_prefix),
                progress_every_elements=progress_every_elements,
            )

            if swapping_fn is not None:
                swapped = swapping_fn(nearest)
                swapped_blocks = (swapped != nearest).sum().item()

            quantized_blocks = motifs[nearest]

        counts = torch.bincount(nearest, minlength=k_eff).to("cpu", dtype=torch.long)
        motif_counts[:k_eff] = counts

        diff = all_blocks - quantized_blocks
        distortion = diff.square().sum().item()
        changed_weights = (all_blocks != quantized_blocks).sum().item()

        offset = 0
        for param, n_blocks, n_params, shape in layer_specs:
            next_offset = offset + n_blocks
            q_blocks = quantized_blocks[offset:next_offset]

            param.data.copy_(blocks_to_tensor(q_blocks, n_params, shape))
            offset = next_offset

    return {
        "distortion": float(distortion),
        "num_changed_weights": int(changed_weights),
        "motif_counts": motif_counts,
        "swapped_blocks": swapped_blocks,
    }


# ------------------------------------------------------------------------------


def quantize_momos(model, quant_cfg):
    """Apply one MoMos projection step and return projection stats."""
    from_percentile = quant_cfg.get("from_percentile", None)
    to_percentile = quant_cfg.get("to_percentile", None)
    probability = quant_cfg.get("swapping_probability", None)

    if from_percentile and to_percentile and probability:
        swapping_function = build_swap_motif(
            from_percentile, to_percentile, probability
        )
    else:
        swapping_function = None

    out = momos(
        model,
        quant_cfg["s"],
        quant_cfg["k"],
        force_zero=quant_cfg.get("force_zero", False),
        chunk_size=quant_cfg.get("chunk_size"),
        show_chunk_progress=quant_cfg.get("chunk_progress", False),
        progress_prefix=quant_cfg.get("progress_prefix", "computing nearest motifs"),
        progress_every_elements=quant_cfg.get("chunk_progress_elements"),
        swapping_fn=swapping_function,
    )
    return out


METHODS = {
    "qat": quantize_qat,
    "momos": quantize_momos,
}


def available_methods():
    """Return supported quantization method names."""
    return sorted(METHODS.keys())


class MoMos:
    """Callable helper that applies MoMos using stored settings."""

    def __init__(
        self,
        s,
        k=None,
        capacity=None,
        q=32,
        force_zero=True,
        chunk_size=None,
        chunk_progress=False,
        chunk_progress_elements=None,
    ):
        """Store MoMos hyperparameters.

        Args:
            s: Block size.
            k: Optional motif count.
            capacity: Optional ratio used to derive motif count.
            q: Optional QAT bit-width metadata (used by callers that also
                enable fake-quant QAT separately).
            force_zero: Whether to force inclusion of zero motif.
            chunk_size: Optional memory budget in MB for nearest-motif distance
                chunking. Default is ``4096`` MB (~4 GB).
            chunk_progress: If True, print coarse chunk progress updates.
            chunk_progress_elements: Optional progress print interval measured
                in processed scalar elements.
        """
        self.s = int(s)
        self.k = None if k is None else int(k)
        self.capacity = capacity
        self.q = int(q)
        self.force_zero = bool(force_zero)
        self.chunk_size = None if chunk_size is None else float(chunk_size)
        self.chunk_progress = bool(chunk_progress)
        self.chunk_progress_elements = (
            None if chunk_progress_elements is None else int(chunk_progress_elements)
        )

    def resolve_k(self, model):
        """Resolve motif count directly or from capacity.

        Args:
            model: Model used when ``capacity`` is provided.

        Returns:
            Integer motif count ``k``.
        """
        if self.k is not None:
            return self.k
        if self.capacity is None:
            raise ValueError("MoMos requires k or capacity")
        return k_from_capacity(model, self.s, self.capacity)

    def config(self, model):
        """Build quantizer config dict for a given model.

        Args:
            model: Model used for resolving ``k`` when needed.

        Returns:
            Config dictionary consumed by ``quantize_momos``.
        """
        return {
            "method": "momos",
            "s": self.s,
            "k": self.resolve_k(model),
            "q": self.q,
            "force_zero": self.force_zero,
            "capacity": self.capacity,
            "chunk_size": self.chunk_size,
            "chunk_progress": self.chunk_progress,
            "chunk_progress_elements": self.chunk_progress_elements,
        }

    def __call__(self, model):
        """Apply MoMos once and return quantization stats.

        Args:
            model: Model to quantize.

        Returns:
            Stats dict including method name.
        """
        out = quantize_momos(model, self.config(model))
        out["method"] = "momos"
        return out


def quantize(model, quant_cfg):
    """Dispatch quantization by method and attach elapsed time.

    Args:
        model: Model to quantize.
        quant_cfg: Quantization config dict with ``method`` key.

    Returns:
        Stats dict with ``method`` and ``q_time``, or ``None``.
    """
    if not quant_cfg:
        return None

    start = time.perf_counter()
    method = str(quant_cfg.get("method", "qat")).lower()
    fn = METHODS.get(method)
    if fn is None:
        raise ValueError(
            f"Unsupported quantization method: {method}. Available: {', '.join(available_methods())}"
        )
    out = fn(model, quant_cfg) or {}
    out["method"] = method
    out["q_time"] = time.perf_counter() - start
    return out
