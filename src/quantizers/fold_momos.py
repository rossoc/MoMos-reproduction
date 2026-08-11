"""Fold MoMos: k-means++ refined codebook via fold-stratified sampling.

Runs a primary MoMos2D pass to get a preliminary motif dictionary, then reuses the
same per-fold (per secondary-grid-position) weighted sampling machinery as
``hierarchical_momos2d`` to build a pooled training set, fits k-means++ once on
that pool to obtain a single *shared* refined codebook, and requantizes the whole
model against it.

Unlike ``hierarchical_momos2d``, there is no per-fold restriction on which motifs
a block may use afterwards -- every block can map to any refined centroid. The
per-fold sampling only shapes *which points feed the k-means fit* (a motif that is
common across many folds gets pooled in repeatedly, which is what pulls the
clustering toward it -- no explicit sample weighting is needed).

Uses ``sklearn.cluster.MiniBatchKMeans`` rather than exact ``KMeans``: this pass
reruns from scratch every validation epoch on a freshly-resampled primary
dictionary (same "no convergence guarantee, heuristic projection" philosophy as
the rest of MoMos -- see ``note/2026-06-15.md``), and at the wide end of the
config space (many folds, large primary capacity) ``n_clusters`` can reach into
the thousands -- exact Lloyd's KMeans scales roughly O(n_samples * n_clusters)
per iteration and was measured at ~6 minutes for a single fit there (14790
clusters, ~490k pooled points); MiniBatchKMeans with a capped ``max_iter`` cuts
that to tens of seconds with no measurable quality loss for this use case.
"""

import torch
from sklearn.cluster import MiniBatchKMeans

from .momos2d import quantize_momos2D, _reconstruct_and_apply
from .momos_hierarchy import _build_fold_grids, _sample_fold_ids
from .block_utils import _nearest_motifs_chunked


def quantize_fold_momos(model, quant_cfg):
    primary_cfg = dict(quant_cfg["primary"])
    primary_cfg["method"] = "momos2d"
    stats = quantize_momos2D(model, primary_cfg) or {}

    Z = stats.pop("_motifs", None)
    M = stats.pop("_nearest", None)
    layer_specs = stats.pop("_layer_specs", None)

    if not quant_cfg.get("apply_secondary", False) or Z is None or M is None or layer_specs is None:
        return stats

    secondary_cfg = dict(quant_cfg["secondary"])
    fold_stats = _fold_kmeans_pass(
        Z=Z,
        M=M,
        layer_specs=layer_specs,
        rows=int(primary_cfg["rows"]),
        cols=int(primary_cfg["cols"]),
        sec_rows=int(secondary_cfg["rows"]),
        sec_cols=int(secondary_cfg["cols"]),
        capacity=float(secondary_cfg.get("capacity", 1.0)),
        force_zero=bool(secondary_cfg.get("force_zero", False)),
        n_init=secondary_cfg.get("kmeans_n_init", "auto"),
        max_iter=int(secondary_cfg.get("kmeans_max_iter", 20)),
    )
    # Internal keys are for direct/test callers of _fold_kmeans_pass only -- keep
    # the public stats dict (logged by QuantizationCallback) scalar-only.
    fold_stats.pop("_refined_motifs", None)
    fold_stats.pop("_refined_mosaic", None)
    return fold_stats


def _fold_kmeans_pass(
    Z,
    M,
    layer_specs,
    rows,
    cols,
    sec_rows,
    sec_cols,
    capacity,
    force_zero,
    n_init,
    max_iter=20,
):
    K = int(Z.shape[0])
    device = Z.device
    dtype = Z.dtype

    with torch.no_grad():
        zero_rows = (Z.abs().sum(dim=1) == 0).nonzero(as_tuple=True)[0]
        zero_idx = int(zero_rows[0].item()) if force_zero and len(zero_rows) else None

        counts, _grids, n_big_total = _build_fold_grids(
            Z, M, layer_specs, rows, cols, sec_rows, sec_cols
        )

        # Same k_per_bucket sizing as hierarchical_momos2d, but here it only
        # controls how many candidate motifs are drawn per fold to build the
        # k-means training pool -- it does not restrict the final codebook range
        # any block may map to.
        k_per_bucket = max(1, min(round(K * capacity), int(n_big_total)))
        iotas = _sample_fold_ids(counts, k_per_bucket, force_zero, zero_idx)

        pooled_ids = (
            torch.cat(iotas) if iotas else torch.empty(0, dtype=torch.long, device=device)
        )

        if pooled_ids.numel() == 0:
            # Nothing to cluster (e.g. no big blocks) -- keep the primary codebook.
            Z_refined = Z.clone()
        else:
            pooled = Z[pooled_ids].detach().cpu().numpy()
            n_clusters = max(1, min(K, pooled.shape[0]))
            # batch_size >= n_clusters so each minibatch has a real chance of
            # updating every cluster; capped at the pool size itself.
            batch_size = max(1, min(pooled.shape[0], max(1024, n_clusters)))
            km = MiniBatchKMeans(
                n_clusters=n_clusters,
                init="k-means++",
                n_init=n_init,
                max_iter=max_iter,
                batch_size=batch_size,
            )
            km.fit(pooled)
            Z_refined = torch.tensor(
                km.cluster_centers_, device=device, dtype=dtype
            )

        if force_zero:
            zero_like = int(Z_refined.norm(dim=1).argmin().item())
            Z_refined[zero_like] = 0.0

        # A block's current value only depends on which primary motif it holds
        # (Z[M]), so mapping every block to its nearest refined centroid reduces
        # to a small K -> K' lookup applied through the mosaic.
        motif_to_centroid = _nearest_motifs_chunked(Z, Z_refined)
        new_M = motif_to_centroid[M]

        primary_blocks = Z[M]
        quantized_blocks = Z_refined[new_M]

        distortion, changed_weights = _reconstruct_and_apply(
            layer_specs, primary_blocks, quantized_blocks, rows, cols
        )

        return {
            "distortion": float(distortion),
            "num_changed_weights": int(changed_weights),
            "num_motifs_used": int(Z_refined.shape[0]),
            # Internal-only, popped by quantize_fold_momos before returning to the
            # training loop -- exposed here so callers/tests can inspect the
            # refined codebook and the post-refinement mosaic directly.
            "_refined_motifs": Z_refined,
            "_refined_mosaic": new_M,
        }
