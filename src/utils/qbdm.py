"""Quantized bit-plane BDM (qbdm) complexity metric.

Extends the sign-only BDM complexity in ``metrics.py`` to multi-bit weight
quantization: each trainable weight matrix is uniformly quantized to
``bit_depth`` levels, decomposed into binary bit-planes, and the Block
Decomposition Method (BDM) algorithmic complexity is summed across all planes
and weight tensors. ``qbdm_complexity`` is the scalar entry point used by
``WeightAnalyzer.qbdm``.
"""

import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torch.nn as nn

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="pkg_resources is deprecated as an API.*",
        category=UserWarning,
        module=r"pybdm\.utils",
    )
    from pybdm import BDM

# Global BDM instance reused by worker processes (2D binary planes).
bdm_instance = BDM(ndim=2, nsymbols=2)


def bdm_batch_worker(data_list):
    """Compute BDM for a batch of binary planes in a single worker call.

    Batching amortizes the per-call inter-process overhead.
    """
    results = []
    for data in data_list:
        try:
            results.append(float(bdm_instance.bdm(data)))
        except Exception:
            results.append(0.0)
    return results


def get_bitplanes(weight_tensor, num_planes, robust=False, percentile=99.9, epsilon=1e-8):
    """Uniformly quantize a weight matrix and return its binary bit-planes.

    Args:
        weight_tensor: Weight tensor; tensors with >2 dims are flattened to 2D.
        num_planes: Number of bits (planes) in the uniform quantizer.
        robust: If True, clip the dynamic range to the central ``percentile``.
        percentile: Central percentile kept when ``robust`` is True.
        epsilon: Guard for (near-)constant tensors.

    Returns:
        List of ``num_planes`` 2D int8 arrays (one per bit), or ``[]`` when the
        matrix is smaller than 4x4 (BDM's 2D window).
    """
    if weight_tensor.dim() > 2:
        weight_tensor = weight_tensor.flatten(1)
    h, w = weight_tensor.shape
    if h < 4 or w < 4:
        return []

    # Determine dynamic range boundaries for the quantizer
    if robust:
        flat_w = weight_tensor.reshape(-1).float()
        q = (1.0 - percentile / 100.0) / 2.0
        w_min = torch.quantile(flat_w, q)
        w_max = torch.quantile(flat_w, 1.0 - q)
    else:
        w_min, w_max = weight_tensor.min(), weight_tensor.max()

    # Quantization step size (Delta)
    delta = (w_max - w_min) / (2 ** num_planes - 1)

    # Map to integer space with uniform rounding and saturation (clamping).
    if (w_max - w_min) < epsilon:
        quantized = torch.zeros_like(weight_tensor, dtype=torch.float32)
    else:
        quantized = torch.clamp(
            torch.round((weight_tensor - w_min) / delta), 0, 2 ** num_planes - 1
        )

    planes = []
    for i in range(num_planes):
        plane = torch.div(quantized, 2 ** i, rounding_mode="floor") % 2
        planes.append(plane.cpu().numpy().astype(np.int8))
    return planes


def get_tiled_manifold(weight_tensor, num_planes, robust=False, percentile=99.9):
    """Construct a 2D tiled manifold interleaving planes to capture coupling."""
    planes = get_bitplanes(weight_tensor, num_planes, robust, percentile)
    if not planes:
        return None

    h, w = planes[0].shape
    if num_planes == 4:
        tr, tc = 2, 2
    elif num_planes == 8:
        tr, tc = 4, 2
    elif num_planes == 2:
        tr, tc = 2, 1
    else:
        tr, tc = num_planes, 1

    tiled = np.zeros((h * tr, w * tc), dtype=np.int8)
    for i in range(tr):
        for j in range(tc):
            bit_idx = i * tc + j
            if bit_idx < num_planes:
                tiled[i::tr, j::tc] = planes[bit_idx]
    return tiled


def measure_complexity(
    model, bit_depths=[8], max_workers=8, robust=False, percentile=99.9, tiled=False
):
    """Aggregate BDM complexity over sign planes and multi-bit bit-planes.

    Accepts an ``nn.Module`` (uses parametrized/quantized weights of trainable
    layers with ``dim >= 2``) or a single 2D+ tensor.

    Returns:
        ``(total_bin, total_multi, total_tiled)`` where ``total_bin`` is the BDM
        over sign-binarized planes, ``total_multi`` maps each bit depth to a list
        of per-plane BDM sums, and ``total_tiled`` maps each bit depth to the
        tiled-manifold BDM (``None`` when ``tiled`` is False).
    """
    binary_tasks = []
    multi_tasks = {bd: [] for bd in bit_depths}
    tiled_tasks = {bd: [] for bd in bit_depths} if tiled else {}
    tensors = []

    # Identify tensors to process: either from a model or a direct tensor input.
    if isinstance(model, nn.Module):
        for name, module in model.named_modules():
            if hasattr(module, "weight"):
                # Property access returns the parametrized (quantized) weight.
                w = module.weight

                # Trainable if the underlying ("original") parameter requires grad.
                is_trainable = False
                if hasattr(module, "parametrizations") and "weight" in module.parametrizations:
                    is_trainable = module.parametrizations.weight.original.requires_grad
                elif isinstance(w, torch.nn.Parameter):
                    is_trainable = w.requires_grad
                if is_trainable and w is not None and w.dim() >= 2:
                    tensors.append(w)
    elif torch.is_tensor(model):
        if model.dim() >= 2:
            tensors.append(model)

    for p_data in tensors:
        p_val = p_data.data
        binary_tasks.append((p_val > 0).cpu().numpy().astype(np.int8))
        for bd in bit_depths:
            if tiled:
                tm = get_tiled_manifold(p_val, bd, robust=robust, percentile=percentile)
                if tm is not None:
                    tiled_tasks[bd].append(tm)

            planes = get_bitplanes(p_val, bd, robust=robust, percentile=percentile)
            multi_tasks[bd].extend(planes)

    def chunk_list(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        bin_chunks = list(chunk_list(binary_tasks, max(1, len(binary_tasks) // max_workers)))
        bin_results = list(executor.map(bdm_batch_worker, bin_chunks))
        total_bin = sum(sum(res) for res in bin_results)

        total_multi = {}
        total_tiled = {} if tiled else None

        for bd in bit_depths:
            m_tasks = multi_tasks[bd]
            m_chunks = list(chunk_list(m_tasks, max(1, len(m_tasks) // (max_workers * 2))))
            m_results = list(executor.map(bdm_batch_worker, m_chunks))

            flat = [score for chunk in m_results for score in chunk]
            total_multi[bd] = [sum(flat[i::bd]) for i in range(bd)]  # re-group by plane

            if tiled:
                t_tasks = tiled_tasks[bd]
                t_chunks = list(chunk_list(t_tasks, max(1, len(t_tasks) // max_workers)))
                t_results = list(executor.map(bdm_batch_worker, t_chunks))
                total_tiled[bd] = sum(sum(res) for res in t_results)

    return float(total_bin), total_multi, total_tiled


def qbdm_complexity(model, bit_depth=8, max_workers=8, robust=False, percentile=99.9):
    """Scalar quantized bit-plane BDM complexity for a model (or 2D+ tensor).

    Quantizes each trainable weight matrix to ``bit_depth`` levels, decomposes it
    into binary bit-planes, and returns the total BDM complexity summed across
    all planes and weight tensors.
    """
    _, total_multi, _ = measure_complexity(
        model,
        bit_depths=[bit_depth],
        max_workers=max_workers,
        robust=robust,
        percentile=percentile,
        tiled=False,
    )
    return float(sum(total_multi[bit_depth]))
