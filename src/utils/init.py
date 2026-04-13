import random
import torch
import numpy as np
import os
from omegaconf import OmegaConf


def generate_unique_run_name(base_dir, slug):
    """Generate a unique run name by prepending an incrementing number to a slug.

    Scans existing directories in ``base_dir`` to find the next available run number.

    Args:
        base_dir: Base directory containing previous run directories.
        slug: Coolname slug string (e.g. ``"foggy-cat"``).

    Returns:
        Unique run name string (e.g. ``"3_foggy-cat"``).
    """
    if not os.path.isdir(base_dir):
        return f"0_{slug}"

    existing_runs = [
        d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
    ]
    numbers = []
    for d in existing_runs:
        parts = d.split("_", 1)
        if parts[0].isdigit():
            numbers.append(int(parts[0]))

    run_number = max(numbers) + 1 if numbers else 0
    return f"{run_number}_{slug}"


def setup_checkpoint_dir(log_dir, dataset_name, prefix, run_name):
    """Create checkpoint directory structure and return paths.

    Args:
        log_dir: Base logging directory.
        dataset_name: Name of the dataset (used in default directory name).
        prefix: Optional prefix for the run directory. If None, defaults to ``"{dataset_name}_mlp"``.
        run_name: Coolname slug for the run (e.g. ``"foggy-cat"``).

    Returns:
        Tuple of ``(checkpoint_dir, unique_run_name, init_ckpt_path)``.
    """
    base_dir = prefix or f"{dataset_name}_mlp"
    run_base_dir = os.path.join(log_dir, base_dir)
    os.makedirs(run_base_dir, exist_ok=True)
    unique_run_name = generate_unique_run_name(run_base_dir, run_name)
    checkpoint_dir = os.path.join(run_base_dir, unique_run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    init_ckpt_path = os.path.join(checkpoint_dir, "init.ckpt")
    return checkpoint_dir, unique_run_name, init_ckpt_path


def normalize_pct(value, field_name):
    """Normalize split value to a fraction in ``(0, 1]``.

    Args:
        value: Ratio in ``(0, 1]``, percent in ``(0, 100]``, or ``None``.
        field_name: Field label used in error messages (e.g. ``val_pct``).

    Returns:
        Normalized fraction, or ``None`` when ``value`` is ``None``.
    """
    if value is None:
        return None
    value = float(value)
    if 1.0 < value <= 100.0:
        value /= 100.0
    if value <= 0.0 or value > 1.0:
        raise ValueError(f"{field_name} must be in (0,1] or (0,100]. Got {value}")
    return value


def seed_all(seed):
    """Seed Python, NumPy, and PyTorch RNGs.

    Args:
        seed: Integer seed used across libraries.
    """
    seed = int(seed)
    random.seed(seed)
    if np is not None:
        np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(mode):
    """Resolve a device mode string to a concrete runtime device.

    Args:
        mode: One of ``auto``, ``cuda``, ``mps``, or ``cpu``.

    Returns:
        Resolved device string.
    """
    mode = str(mode).lower()
    has_mps = (
        bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
    )

    if mode == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "mps" if has_mps else "cpu"
    elif mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA but it is not available.")
        return "cuda"
    elif mode == "mps":
        if not has_mps:
            raise RuntimeError("Requested MPS but it is not available.")
        return "mps"
    elif mode == "cpu":
        return "cpu"
    raise ValueError(f"Unknown device mode: {mode}")


def resolve_runtime(mode):
    accelerator = resolve_device(mode)

    if accelerator in ("cuda", "mps", "cpu"):
        runtime_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "configs",
            "runtime",
            f"{accelerator}.yaml",
        )
        if os.path.exists(runtime_path):
            runtime_cfg = OmegaConf.load(runtime_path)

    return accelerator, runtime_cfg


def configure_cuda_fast_path():
    """Enable common CUDA fast-path settings when CUDA is available."""
    if not torch.cuda.is_available():
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def format_seconds(seconds):
    """Format seconds as ``MM:SS`` or ``HH:MM:SS``.

    Args:
        seconds: Elapsed time in seconds.

    Returns:
        Human-readable duration string.
    """
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"
