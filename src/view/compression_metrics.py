def _compute_rac(run):
    """
    Compute Relative Algorithmic Compression (RAC) rate.

    RAC = K(W)/K(What) ≈ nq/(ksq + m*ceil(log2(k)))
    For QAT, report as 32/q relative to FP32.

    Parameters:
    -----------
    run : dict
        Experiment run with config and metrics

    Returns:
    --------
    float
        RAC rate (lower is better = more compression)
    """
    config = run.get("config", {})
    metrics = run.get("metrics", {})

    s = config.get("s", 1)  # block size
    k = config.get("k", 1)  # number of motifs
    q = config.get("q", 16)  # quantization bits

    # Get model parameters from config or defaults
    n = config.get("n", 256)  # number of neurons
    m = config.get("m", 2)  # number of motifs per layer

    # Compute RAC using the formula: nq / (k*s*q + m*ceil(log2(k)))
    # For MoMos: numerator is the compressed representation size
    # denominator is the description length of the MoMos construction
    import math

    denominator = k * s * q + m * math.ceil(math.log2(k))
    rac = (n * q) / denominator if denominator > 0 else float("inf")

    # For QAT, report as 32/q relative to FP32
    rac_qat = 32.0 / q if q and q != 16 else None

    return rac, rac_qat


def _extract_bdm_complexity_from_wandb(
    entity, project, run_name=None, artifact_path=""
):
    """
    Extract BDM complexity metrics from wandb artifacts.

    BDM ratio = K_B(W*)/K_B(W0) where weights are binarized.
    The metrics already exist in wandb: metrics/bdm_complexity

    Parameters:
    -----------
    entity : str
        WandB entity name
    project : str
        WandB project name
    run_name : str, optional
        Specific run name to fetch
    artifact_path : str, optional
        Path to artifact in wandb

    Returns:
    --------
    tuple
        (bdm_complexity, bdm_ratio) or (None, None) if not found
    """
    import wandb
    from wandb.apis.public import Runs

    api = wandb.Api()

    # Try to get run info
    runs = api.runs(entity + "/" + project, lazy=False)

    target_run = None
    if run_name:
        for run in runs:
            if run_name in run.name or run_name in run.id:
                target_run = run
                break

    if not target_run:
        # Try to get from first available run
        target_run = runs[0] if len(runs) > 0 else None

    if not target_run:
        return None, None

    # Get run history for metrics
    try:
        history = target_run.history()
        bdm_complexity = history.get("metrics/bdm_complexity")
        bdm_complexity = float(bdm_complexity) if bdm_complexity is not None else None
    except Exception:
        bdm_complexity = None

    # Try to get BDM ratio from config or metrics
    config = target_run.config
    bdm_ratio = config.get("bdm_ratio") or config.get("bdm_complexity_ratio")

    return bdm_complexity, bdm_ratio


def _fetch_bdm_from_artifact(artifact_id):
    """
    Fetch BDM complexity from a specific wandb artifact.

    Parameters:
    -----------
    artifact_id : str
        Full artifact ID including entity/project/path/version

    Returns:
    --------
    dict
        Dictionary with bdm_complexity and bdm_ratio
    """
    import wandb

    api = wandb.Api()

    try:
        artifact = api.artifact(artifact_id, type="model")
        artifact_dir = artifact.download()

        # Look for metrics file in artifact
        import os

        metrics_file = None
        for root, dirs, files in os.walk(artifact_dir):
            for f in files:
                if "metrics" in f.lower() or "bdm" in f.lower():
                    metrics_file = os.path.join(root, f)
                    break
            if metrics_file:
                break

        # Parse metrics from artifact metadata
        result = {}
        if metrics_file and os.path.exists(metrics_file):
            import json

            with open(metrics_file, "r") as f:
                metrics = json.load(f)
                result.update(metrics)

        return result
    except Exception as e:
        print(f"Error fetching BDM from artifact: {e}")
        return {}


def compute_compression_metrics(run):
    """
    Compute both RAC and BDM complexity metrics for a run.

    Parameters:
    -----------
    run : dict
        Experiment run with config and metrics

    Returns:
    --------
    dict
        Dictionary with rac, rac_qat, bdm_complexity, bdm_ratio
    """
    import math

    config = run.get("config", {})
    metrics = run.get("metrics", {})

    # Compute RAC
    s = config.get("s", 1)
    k = config.get("k", 1)
    q = config.get("q", 16)
    n = config.get("n", 256)
    m = config.get("m", 2)

    denominator = k * s * q + m * math.ceil(math.log2(k)) if k > 0 else 1
    rac = (n * q) / denominator if denominator > 0 else float("inf")

    # For QAT, report as 32/q relative to FP32
    rac_qat = 32.0 / q if q and q != 16 else None

    # Extract BDM complexity from metrics
    bdm_complexity = metrics.get("metrics/bdm_complexity")
    if bdm_complexity is None:
        bdm_complexity = metrics.get("bdm_complexity")
    bdm_complexity = float(bdm_complexity) if bdm_complexity is not None else None

    # Try to get BDM ratio
    bdm_ratio = metrics.get("metrics/bdm_complexity_ratio")
    if bdm_ratio is None:
        bdm_ratio = metrics.get("bdm_complexity_ratio")
    bdm_ratio = float(bdm_ratio) if bdm_ratio is not None else None

    return {
        "rac": rac,
        "rac_qat": rac_qat,
        "bdm_complexity": bdm_complexity,
        "bdm_ratio": bdm_ratio,
    }
