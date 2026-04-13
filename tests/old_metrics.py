from src.utils.metrics import WeightAnalyzer, compression_rate
import bz2
import gzip
import lzma


def compute_sparsity(model, compression_binarized=False):
    """Compute exact-zero fraction over trainable weights.

    Args:
        model: PyTorch model to inspect.
        compression_binarized: Unused; kept for metric API consistency.

    Returns:
        Dict containing ``sparsity``.
    """
    _ = compression_binarized
    analyzer = WeightAnalyzer(model)
    return {"sparsity": analyzer.sparsity()}


def compute_l2(model, compression_binarized=False):
    """Compute L2 norm over trainable weights.

    Args:
        model: PyTorch model to inspect.
        compression_binarized: Unused; kept for metric API consistency.

    Returns:
        Dict containing ``weight_l2``.
    """
    _ = compression_binarized
    analyzer = WeightAnalyzer(model)
    return {"weight_l2": analyzer.l2_norm()}


def compute_bdm(model, compression_binarized=False):
    """Compute BDM complexity on binarized model weights.

    Args:
        model: PyTorch model to inspect.
        compression_binarized: Unused; BDM always uses binarized payload.

    Returns:
        Dict containing ``bdm_complexity`` (or ``None`` if unavailable).
    """
    _ = compression_binarized
    analyzer = WeightAnalyzer(model)
    return {"bdm_complexity": analyzer.bdm_complexity()}


def compute_gzip(model, compression_binarized=False):
    """Compute gzip compression rate of model weights.

    Args:
        model: PyTorch model to inspect.
        compression_binarized: If True, compress sign bits only.

    Returns:
        Dict containing ``gzip_compression_rate``.
    """
    analyzer = WeightAnalyzer(model)
    payload = analyzer.get_payload(binarized=compression_binarized)
    return {
        "gzip_compression_rate": compression_rate(
            payload, gzip.compress(payload, compresslevel=9, mtime=0)
        )
    }


def compute_bz2(model, compression_binarized=False):
    """Compute bz2 compression rate of model weights.

    Args:
        model: PyTorch model to inspect.
        compression_binarized: If True, compress sign bits only.

    Returns:
        Dict containing ``bz2_compression_rate``.
    """
    analyzer = WeightAnalyzer(model)
    payload = analyzer.get_payload(binarized=compression_binarized)
    return {
        "bz2_compression_rate": compression_rate(
            payload, bz2.compress(payload, compresslevel=9)
        )
    }


def compute_lzma(model, compression_binarized=False):
    """Compute lzma compression rate of model weights.

    Args:
        model: PyTorch model to inspect.
        compression_binarized: If True, compress sign bits only.

    Returns:
        Dict containing ``lzma_compression_rate``.
    """
    analyzer = WeightAnalyzer(model)
    payload = analyzer.get_payload(binarized=compression_binarized)
    return {
        "lzma_compression_rate": compression_rate(
            payload, lzma.compress(payload, preset=9)
        )
    }


def get_compression_payload(model, compression_binarized):
    """Serialize model weights to bytes for compression metrics.

    Args:
        model: PyTorch model to inspect.
        compression_binarized: If True, serialize sign bits only.

    Returns:
        Byte payload used by compression backends.
    """
    analyzer = WeightAnalyzer(model)
    return analyzer.get_payload(binarized=compression_binarized)


registry = {
    "sparsity": compute_sparsity,
    "l2": compute_l2,
    "bdm": compute_bdm,
    "gzip": compute_gzip,
    "bz2": compute_bz2,
    "lzma": compute_lzma,
}
