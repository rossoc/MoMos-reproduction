"""MoMos 2D quantizer with G-function magnitude encoding.

Pipeline:
  1. Random motif initialisation (standard MoMos — sample actual blocks).
  2. First assignment pass — nearest-motif for each block.
  3. G(i) encode — snap each motif's mean magnitude to the closest codebook
     entry G(i), preserving element-wise signs.
  4. Second assignment pass — reassign blocks to the encoded motifs.

Three G(i) modes are supported:
  exp_inverse  — G(i) = A·(exp(B·i)−1), analytic inverse available
  exp_lookup   — same G(i), encoding via searchsorted over precomputed table
  sr_rational  — SR rational function, R²=0.9924 (best quality)

Default G-function parameters are fitted on a v20 CIFAR-10/MLP checkpoint.
"""

import torch
from .block_utils import build_swap_motif, _assign_blocks, _initialize_motifs
from .momos2d import blocks_to_tensor2D, _get_model_blocks2D

# ---------------------------------------------------------------------------
# G(i) functions: index → weight magnitude
# ---------------------------------------------------------------------------

_EXP_DEFAULTS = dict(A=7.014e-03, B=1.185e-02)


def g_exp(i: torch.Tensor, A: float, B: float) -> torch.Tensor:
    """Exponential magnitude codebook: G(i) = A·(exp(B·i)−1)."""
    return A * (torch.exp(B * i) - 1.0)


def g_exp_inverse(w: torch.Tensor, A: float, B: float) -> torch.Tensor:
    """Analytic inverse of g_exp: maps |w| → codebook index i."""
    return torch.log(w.abs() / A + 1.0) / B


def g_sr_rational(i: torch.Tensor) -> torch.Tensor:
    """SR rational magnitude codebook (PySR best expression).

    G(i) = 0.00033626049238844·i / (66.02802 − 0.2584038·i)^(1/4)
    """
    return 0.00033626049238844 * i / (66.02802 - 0.2584038 * i).pow(0.25)


# ---------------------------------------------------------------------------
# Codebook precomputation (256 entries)
# ---------------------------------------------------------------------------


def build_codebook(init_mode: str, A: float, B: float) -> torch.Tensor:
    """Return the 256-entry magnitude codebook for the given init_mode."""
    i = torch.arange(256, dtype=torch.float32)
    if init_mode == "sr_rational":
        return g_sr_rational(i)
    return g_exp(i, A, B)


# ---------------------------------------------------------------------------
# Static motif initialisation
# ---------------------------------------------------------------------------


def _initialize_motifs_static(
    k_eff: int,
    block_size: int,
    init_mode: str,
    force_zero: bool,
    A: float,
    B: float,
    device: torch.device,
) -> torch.Tensor:
    """Build initial motif matrix from G(i) instead of random block sampling.

    Samples k_eff evenly-spaced magnitudes from the 256-entry codebook and
    applies random signs to each block element.  If force_zero is True, the
    first motif is the zero vector and G-derived motifs fill slots [1, k_eff).
    """
    codebook = build_codebook(init_mode, A, B).to(device)  # (256,)

    n_derived = k_eff - int(force_zero)
    if n_derived <= 0:
        return torch.zeros(k_eff, block_size, device=device)

    idx = torch.linspace(0, 255, n_derived, device=device).long()
    magnitudes = codebook[idx]  # (n_derived,)

    signs = torch.randint(0, 2, (n_derived, block_size), device=device) * 2 - 1
    derived = signs * magnitudes.unsqueeze(1)  # (n_derived, block_size)

    if force_zero:
        motifs = torch.zeros(k_eff, block_size, device=device)
        motifs[1:] = derived
    else:
        motifs = derived

    return motifs


# ---------------------------------------------------------------------------
# G(i) magnitude encoding
# ---------------------------------------------------------------------------


def _encode_motifs_g(
    motifs: torch.Tensor,
    codebook: torch.Tensor,
    force_zero: bool,
) -> torch.Tensor:
    """Snap each motif's magnitude to the nearest G(i) codebook entry.

    Computes the per-motif mean absolute value, finds the closest entry in the
    monotone codebook via searchsorted, and replaces every element with
    ±G(i*) (element-wise sign preserved).  The zero motif (slot 0 when
    force_zero=True) is left untouched.
    """
    start = int(force_zero)
    if start >= motifs.size(0):
        return motifs

    subset = motifs[start:]  # (n, block_size)
    mean_abs = subset.abs().mean(dim=1)  # (n,)

    idx = torch.searchsorted(codebook.contiguous(), mean_abs.contiguous())
    idx = idx.clamp(0, len(codebook) - 1)

    idx_prev = (idx - 1).clamp(0)
    d_curr = (codebook[idx] - mean_abs).abs()
    d_prev = (codebook[idx_prev] - mean_abs).abs()
    idx = torch.where(d_prev < d_curr, idx_prev, idx)

    new_mag = codebook[idx].unsqueeze(1)  # (n, 1)

    out = motifs.clone()
    out[start:] = subset.sign() * new_mag
    return out


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------


def static_momos2D(
    model,
    rows: int,
    cols: int,
    k: int,
    init_mode: str = "sr_rational",
    A: float = 9.10e-3,
    B: float = 6.86e-3,
    force_zero: bool = True,
    chunk_size=None,
    show_chunk_progress: bool = False,
    progress_prefix: str = "static_momos",
    progress_every_elements=None,
    swapping_fn=None,
) -> dict:
    """MoMos 2D with random init followed by G(i) magnitude encoding.

    Two-pass pipeline:
      Pass 1 — random motif init (sample actual blocks) + nearest-motif
               assignment, same as standard MoMos.
      Encode  — snap each motif's mean magnitude to the nearest G(i) codebook
               entry, preserving element-wise signs.
      Pass 2 — reassign blocks to the encoded motifs (with optional swapping).

    Args:
        model: Model whose trainable parameters are quantized in-place.
        rows: Block height.
        cols: Block width.
        k: Number of motifs (codebook entries).
        init_mode: One of ``"exp_inverse"``, ``"exp_lookup"``, ``"sr_rational"``.
        A, B: Exponential G-function parameters (used when init_mode is
            ``"exp_inverse"`` or ``"exp_lookup"``).
        force_zero: If True the first motif is the all-zero vector.
        chunk_size: Memory budget in MB for distance computation (default 4096).
        show_chunk_progress: Print progress during assignment.
        progress_prefix: Label for progress output.
        progress_every_elements: Reporting granularity (default ~20 reports).
        swapping_fn: Optional motif-swapping function from build_swap_motif.

    Returns:
        Dict with ``distortion``, ``num_changed_weights``, ``motif_counts``,
        ``swapped_blocks``, and ``init_mode``.
    """
    _VALID_MODES = ("exp_inverse", "exp_lookup", "sr_rational")
    if init_mode not in _VALID_MODES:
        raise ValueError(f"init_mode must be one of {_VALID_MODES}, got {init_mode!r}")

    rows, cols, k = int(rows), int(cols), int(k)
    block_size = rows * cols
    motif_counts = torch.zeros(max(1, k), dtype=torch.long)

    with torch.no_grad():
        all_blocks, layer_specs = _get_model_blocks2D(model, rows, cols)

        if all_blocks is None:
            return {
                "distortion": 0.0,
                "num_changed_weights": 0,
                "motif_counts": motif_counts,
                "swapped_blocks": 0,
                "init_mode": init_mode,
            }

        total_blocks = all_blocks.size(0)
        k_eff = max(1, min(k, total_blocks))

        # Pass 1: random init + assignment
        motifs = _initialize_motifs(all_blocks, k_eff, block_size, force_zero)
        _assign_blocks(
            all_blocks,
            motifs,
            chunk_size,
            show_chunk_progress,
            progress_prefix,
            progress_every_elements,
            None,  # no swapping on first pass
        )

        # Encode: snap motif magnitudes to G(i) codebook
        codebook = build_codebook(init_mode, A, B).to(all_blocks.device)
        motifs = _encode_motifs_g(motifs, codebook, force_zero)

        # Pass 2: reassign with encoded motifs (swapping applied here)
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


# ---------------------------------------------------------------------------
# Config dispatcher
# ---------------------------------------------------------------------------


def quantize_static_momos2D(model, quant_cfg: dict) -> dict:
    """Apply one static MoMos 2D projection step from a config dict."""
    from_percentile = quant_cfg.get("from_percentile", None)
    to_percentile = quant_cfg.get("to_percentile", None)
    probability = quant_cfg.get("swapping_probability", None)

    if from_percentile and to_percentile and probability:
        swapping_function = build_swap_motif(
            from_percentile, to_percentile, probability
        )
    else:
        swapping_function = None

    return static_momos2D(
        model,
        rows=quant_cfg["rows"],
        cols=quant_cfg["cols"],
        k=quant_cfg["k"],
        init_mode=quant_cfg.get("init_mode", "sr_rational"),
        A=quant_cfg.get("exp_A", _EXP_DEFAULTS["A"]),
        B=quant_cfg.get("exp_B", _EXP_DEFAULTS["B"]),
        force_zero=quant_cfg.get("force_zero", True),
        chunk_size=quant_cfg.get("chunk_size"),
        show_chunk_progress=quant_cfg.get("chunk_progress", False),
        progress_prefix=quant_cfg.get("progress_prefix", "static_momos"),
        progress_every_elements=quant_cfg.get("chunk_progress_elements"),
        swapping_fn=swapping_function,
    )
