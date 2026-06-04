"""Core MoMos quantization algorithm, dispatcher, and MoMos class."""

import torch

from .block_utils import (
    iter_trainable_params,
    tensor_to_blocks,
    blocks_to_tensor,
    build_swap_motif,
    _initialize_motifs,
    _assign_blocks,
)


def _get_model_blocks(model, block_size):
    """Iterates through params and converts them to blocks."""
    layer_specs = []
    all_blocks = []
    for param in iter_trainable_params(model):
        blocks, n_params, shape = tensor_to_blocks(param.detach(), block_size)
        layer_specs.append((param, int(blocks.size(0)), int(n_params), shape))
        all_blocks.append(blocks)

    if not all_blocks:
        return None, []

    return torch.cat(all_blocks, dim=0), layer_specs


def _update_model_parameters(layer_specs, quantized_blocks):
    """Reconstructs tensors and copies data back to the model."""
    offset = 0
    for param, n_blocks, n_params, shape in layer_specs:
        next_offset = offset + n_blocks
        q_blocks = quantized_blocks[offset:next_offset]
        param.data.copy_(blocks_to_tensor(q_blocks, n_params, shape))
        offset = next_offset


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
    block_size, k = int(block_size), int(k)
    motif_counts = torch.zeros(max(1, k), dtype=torch.long)

    with torch.no_grad():
        all_blocks, layer_specs = _get_model_blocks(model, block_size)

        if all_blocks is None:
            return {
                "distortion": 0.0,
                "num_changed_weights": 0,
                "motif_counts": motif_counts,
            }

        total_blocks = all_blocks.size(0)
        k_eff = max(1, min(k, total_blocks))

        motifs = _initialize_motifs(all_blocks, k_eff, block_size, force_zero)

        nearest, swapped_blocks = _assign_blocks(
            all_blocks,
            motifs,
            chunk_size,
            show_chunk_progress,
            progress_prefix,
            progress_every_elements,
            swapping_fn,
        )

        quantized_blocks = motifs[nearest]

        # Statistics
        counts = torch.bincount(nearest, minlength=k_eff).to("cpu", dtype=torch.long)
        motif_counts[:k_eff] = counts

        changed_weights = 0
        distortion = 0.0
        offset = 0
        for _param, n_blocks, n_params, _shape in layer_specs:
            next_offset = offset + n_blocks
            layer_orig = all_blocks[offset:next_offset].flatten()[:n_params]
            layer_quant = quantized_blocks[offset:next_offset].flatten()[:n_params]
            distortion += (layer_orig - layer_quant).square().sum().item()
            changed_weights += (layer_orig != layer_quant).sum().item()
            offset = next_offset

        _update_model_parameters(layer_specs, quantized_blocks)

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
