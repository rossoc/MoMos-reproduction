from .block_utils import iter_trainable_params, build_swap_motif
from .momos import _initialize_motifs, _assign_blocks
import torch.nn.functional as F
import torch


def tensor2D_to_blocks(tensor, rows, cols):
    """Flatten and pad a tensor into fixed-size blocks based on last 2 dims."""
    rows, cols = int(rows), int(cols)
    if rows <= 0 or cols <= 0:
        raise ValueError(f"rows and cols must be > 0, ({rows}, {cols})")

    original_shape = tensor.shape
    tensor_view = tensor.view(-1, original_shape[-2], original_shape[-1])
    batch, orig_h, orig_w = tensor_view.shape

    pad_h = (-orig_h) % rows
    pad_w = (-orig_w) % cols

    if pad_h > 0 or pad_w > 0:
        tensor_view = F.pad(tensor_view, (0, pad_w, 0, pad_h))

    new_h, new_w = tensor_view.shape[1], tensor_view.shape[2]

    blocks = tensor_view.view(batch, new_h // rows, rows, new_w // cols, cols)
    blocks = blocks.permute(0, 1, 3, 2, 4)

    return blocks.reshape(-1, rows * cols), tensor.numel(), original_shape


def blocks_to_tensor2D(blocks, original_shape, rows, cols):
    orig_h, orig_w = original_shape[-2], original_shape[-1]
    leading_dims = original_shape[:-2]
    batch = 1
    for d in leading_dims:
        batch *= d

    num_blocks_h = (orig_h + rows - 1) // rows
    num_blocks_w = (orig_w + cols - 1) // cols

    out = blocks.view(batch, num_blocks_h, num_blocks_w, rows, cols)
    out = out.permute(0, 1, 3, 2, 4)

    padded_h = num_blocks_h * rows
    padded_w = num_blocks_w * cols

    out = out.reshape(batch, padded_h, padded_w)
    out = out[:, :orig_h, :orig_w]

    return out.view(original_shape)


def _get_model_blocks2D(model, rows, cols):
    """Iterates through params and converts them to blocks."""
    layer_specs = []
    all_blocks = []
    for param in iter_trainable_params(model):
        blocks, n_params, shape = tensor2D_to_blocks(param.detach(), rows, cols)
        layer_specs.append((param, int(blocks.size(0)), int(n_params), shape))
        all_blocks.append(blocks)

    if not all_blocks:
        return None, []

    return torch.cat(all_blocks, dim=0), layer_specs


def momos2D(
    model,
    rows,
    cols,
    k,
    force_zero=True,
    chunk_size=None,
    show_chunk_progress=False,
    progress_prefix="momos",
    progress_every_elements=None,
    swapping_fn=None,
):
    rows, cols, k = int(rows), int(cols), int(k)
    motif_counts = torch.zeros(max(1, k), dtype=torch.long)

    with torch.no_grad():
        all_blocks, layer_specs = _get_model_blocks2D(model, rows, cols)

        if all_blocks is None:
            return {
                "distortion": 0.0,
                "num_changed_weights": 0,
                "motif_counts": motif_counts,
            }

        total_blocks = all_blocks.size(0)
        k_eff = max(1, min(k, total_blocks))

        motifs = _initialize_motifs(all_blocks, k_eff, rows * cols, force_zero)

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

        diff = all_blocks - quantized_blocks
        distortion = diff.square().sum().item()
        changed_weights = (all_blocks != quantized_blocks).sum().item()

        offset = 0
        for param, n_blocks, n_params, shape in layer_specs:
            next_offset = offset + n_blocks
            q_blocks = quantized_blocks[offset:next_offset]
            param.data.copy_(blocks_to_tensor2D(q_blocks, shape, rows, cols))
            offset = next_offset

    return {
        "distortion": float(distortion),
        "num_changed_weights": int(changed_weights),
        "motif_counts": motif_counts,
        "swapped_blocks": swapped_blocks,
    }


def quantize_momos2D(model, quant_cfg):
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

    out = momos2D(
        model,
        quant_cfg["cols"],
        quant_cfg["rows"],
        quant_cfg["k"],
        force_zero=quant_cfg.get("force_zero", False),
        chunk_size=quant_cfg.get("chunk_size"),
        show_chunk_progress=quant_cfg.get("chunk_progress", False),
        progress_prefix=quant_cfg.get("progress_prefix", "computing nearest motifs"),
        progress_every_elements=quant_cfg.get("chunk_progress_elements"),
        swapping_fn=swapping_function,
    )
    return out
