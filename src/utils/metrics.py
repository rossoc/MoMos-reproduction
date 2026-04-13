import bz2
import gzip
import lzma
import numpy as np
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="pkg_resources is deprecated as an API.*",
        category=UserWarning,
        module=r"pybdm\.utils",
    )
    try:
        from pybdm import BDM

        bdm_engine = BDM(ndim=1)

    except Exception:
        BDM = None  # type: ignore
        bdm_engine = None


def flatten_weights(model):
    """Flatten trainable parameters into one NumPy vector.

    Args:
        model: PyTorch model to inspect.

    Returns:
        1D float32 NumPy array of trainable parameters.
    """
    arrays = []
    for param in model.parameters():
        if param.requires_grad and param.numel() > 0:
            arrays.append(param.detach().cpu().reshape(-1).float().numpy())
    if not arrays:
        return np.array([], dtype=np.float32)
    return np.concatenate(arrays)


def get_compression_payload_from_weights(weights, compression_binarized):
    """Serialize an already flattened weight vector to bytes."""
    if weights.size == 0:
        return b""
    if compression_binarized:
        bits = np.ascontiguousarray((weights > 0).astype(np.uint8))
        return bits.tobytes()
    values = np.ascontiguousarray(weights.astype(np.float32))
    return values.tobytes()


def compression_rate(payload, compressed_payload):
    """Compute compression ratio as raw_size/compressed_size.

    Args:
        payload: Uncompressed bytes.
        compressed_payload: Compressed bytes.

    Returns:
        Float compression ratio (0.0 for empty payload).
    """
    raw_size = len(payload)
    if raw_size == 0:
        return 0.0
    compressed_size = max(1, len(compressed_payload))
    return float(raw_size / compressed_size)


def compute_metrics(model, names, compression_binarized=False):
    """Compute selected metrics on the current model weights.

    Uses WeightAnalyzer internally for efficient caching.

    Args:
        model: PyTorch model to inspect.
        names: List of metric names to compute.
        compression_binarized: Global flag for compression metric payload format.

    Returns:
        Dict of merged metric outputs.
    """
    analyzer = WeightAnalyzer(model, compression_binarized=compression_binarized)
    registry = {
        "sparsity": ("sparsity", analyzer.sparsity),
        "l2": ("weight_l2", analyzer.l2_norm),
        "bdm": ("bdm_complexity", analyzer.bdm_complexity),
        "gzip": ("gzip_compression_rate", analyzer.gzip_compress),
        "bz2": ("bz2_compression_rate", analyzer.bz2_compress),
        "lzma": ("lzma_compression_rate", analyzer.lzma_compress),
    }

    unknown = [n for n in names if n not in registry]
    if unknown:
        available = ", ".join(sorted(registry))
        raise ValueError(
            f"Unknown metric(s): {', '.join(sorted(set(str(n) for n in unknown)))}. Available: {available}"
        )

    return {
        output_key: compute_fn()
        for name in names
        for output_key, compute_fn in [registry[name]]
    }


class WeightAnalyzer:
    """Analyzes model weights with efficient caching.

    Flattens weights once and caches payloads for repeated access.

    Args:
        model: PyTorch model to analyze.
        compression_binarized: If True, use binarized payload for compression metrics.
    """

    def __init__(self, model, compression_binarized=False):
        self.weights = flatten_weights(model)
        self._payload_cache = {}
        self._compression_binarized = compression_binarized

    def set_compression_binarized(self, value):
        """Set the binarized flag for compression payloads.

        Args:
            value: Boolean flag.
        """
        self._compression_binarized = value

    def get_payload(self, binarized):
        """Get compression payload, computing and caching if needed.

        Args:
            binarized: If True, use binarized payload (sign bits only).

        Returns:
            Byte payload for compression.
        """
        if binarized not in self._payload_cache:
            self._payload_cache[binarized] = get_compression_payload_from_weights(
                self.weights, compression_binarized=binarized
            )
        return self._payload_cache[binarized]

    def sparsity(self):
        """Compute exact-zero fraction over trainable weights."""
        if self.weights.size == 0:
            return 0.0
        return float((self.weights == 0).mean())

    def l2_norm(self):
        """Compute L2 norm over trainable weights."""
        if self.weights.size == 0:
            return 0.0
        return float(np.linalg.norm(self.weights))

    def bdm_complexity(self):
        """Compute BDM complexity on binarized model weights.

        Returns:
            Float complexity value, or None if BDM unavailable.
        """
        if bdm_engine is None:
            return None
        if self.weights.size == 0:
            return 0.0
        bits = np.ascontiguousarray((self.weights > 0).astype(np.uint8))
        try:
            return float(bdm_engine.bdm(bits))
        except Exception:
            return None

    def _compress_payload(self, backend):
        """Compress the configured payload and return the compression rate."""
        payload = self.get_payload(binarized=self._compression_binarized)
        if backend == "gzip":
            compressed = gzip.compress(payload, compresslevel=9, mtime=0)
        elif backend == "bz2":
            compressed = bz2.compress(payload, compresslevel=9)
        elif backend == "lzma":
            compressed = lzma.compress(payload, preset=9)
        else:
            raise ValueError(f"Unknown compression backend: {backend}")
        return compression_rate(payload, compressed)

    def gzip_compress(self):
        """Compute gzip compression rate."""
        return self._compress_payload("gzip")

    def bz2_compress(self):
        """Compute bz2 compression rate."""
        return self._compress_payload("bz2")

    def lzma_compress(self):
        """Compute lzma compression rate."""
        return self._compress_payload("lzma")
