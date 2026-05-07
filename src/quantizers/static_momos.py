"""Static (G-function) initialised MoMos 2D quantizer.

Three init modes share identical nearest-motif assignment infrastructure;
only the codebook initialisation differs:

  exp_inverse  — G(i) = A·(exp(B·i^C)−1), analytic inverse available
  exp_lookup   — same G(i), encoding via searchsorted over precomputed table
  sr_rational  — SR rational function, R²=0.9924 (best quality)

Default G-function parameters are fitted on a v20 CIFAR-10/MLP checkpoint.
"""

import torch
from .block_utils import build_swap_motif
from .momos import _assign_blocks
from .momos2d import tensor2D_to_blocks, blocks_to_tensor2D, _get_model_blocks2D

# ---------------------------------------------------------------------------
# G(i) functions: index → weight magnitude
# ---------------------------------------------------------------------------

_EXP_DEFAULTS = dict(A=9.10e-3, B=6.86e-3, C=1.084)


def g_exp(i: torch.Tensor, A: float = 9.10e-3, B: float = 6.86e-3, C: float = 1.084) -> torch.Tensor:
    """3-parameter exponential magnitude codebook: G(i) = A·(exp(B·i^C)−1)."""
    return A * (torch.exp(B * i.pow(C)) - 1.0)


def g_exp_inverse(w: torch.Tensor, A: float = 9.10e-3, B: float = 6.86e-3, C: float = 1.084) -> torch.Tensor:
    """Analytic inverse of g_exp: maps |w| → codebook index i."""
    return (torch.log(w.abs() / A + 1.0) / B).pow(1.0 / C)


def g_sr_rational(i: torch.Tensor) -> torch.Tensor:
    """SR rational magnitude codebook (PySR best expression, v20 checkpoint).

    G(i) = 4.217787 / ((0.62344253 − 0.8370065/(0.3184213 − 0.026600074·i)) · (288.77637−i))
    """
    inner = 0.3184213 - 0.026600074 * i
    denom = (0.62344253 - 0.8370065 / inner) * (288.77637 - i)
    return 4.217787 / denom


# ---------------------------------------------------------------------------
# Codebook precomputation (256 entries)
# ---------------------------------------------------------------------------

def build_codebook(init_mode: str, A: float = 9.10e-3, B: float = 6.86e-3, C: float = 1.084) -> torch.Tensor:
    """Return the 256-entry magnitude codebook for the given init_mode."""
    i = torch.arange(256, dtype=torch.float32)
    if init_mode == "sr_rational":
        return g_sr_rational(i)
    return g_exp(i, A, B, C)


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
    C: float,
    device: torch.device,
) -> torch.Tensor:
    """Build initial motif matrix from G(i) instead of random block sampling.

    Samples k_eff evenly-spaced magnitudes from the 256-entry codebook and
    applies random signs to each block element.  If force_zero is True, the
    first motif is the zero vector and G-derived motifs fill slots [1, k_eff).
    """
    codebook = build_codebook(init_mode, A, B, C).to(device)  # (256,)

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
    C: float = 1.084,
    force_zero: bool = True,
    chunk_size=None,
    show_chunk_progress: bool = False,
    progress_prefix: str = "static_momos",
    progress_every_elements=None,
    swapping_fn=None,
) -> dict:
    """MoMos 2D with G-function codebook initialisation.

    Identical to momos2D except motifs are seeded from a closed-form
    magnitude function G(i) rather than random block sampling.

    Args:
        model: Model whose trainable parameters are quantized in-place.
        rows: Block height.
        cols: Block width.
        k: Number of motifs (codebook entries).
        init_mode: One of ``"exp_inverse"``, ``"exp_lookup"``, ``"sr_rational"``.
        A, B, C: Exponential G-function parameters (used when init_mode is
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

        motifs = _initialize_motifs_static(
            k_eff, block_size, init_mode, force_zero, A, B, C, all_blocks.device
        )

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
        "init_mode": init_mode,
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
        swapping_function = build_swap_motif(from_percentile, to_percentile, probability)
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
        C=quant_cfg.get("exp_C", _EXP_DEFAULTS["C"]),
        force_zero=quant_cfg.get("force_zero", True),
        chunk_size=quant_cfg.get("chunk_size"),
        show_chunk_progress=quant_cfg.get("chunk_progress", False),
        progress_prefix=quant_cfg.get("progress_prefix", "static_momos"),
        progress_every_elements=quant_cfg.get("chunk_progress_elements"),
        swapping_fn=swapping_function,
    )
