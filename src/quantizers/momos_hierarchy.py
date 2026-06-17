"""Hierarchical MoMos2D quantization.

Phase 1 (epoch progress < ``switch_fraction``): standard ``momos2D`` pass.

Phase 2: after the primary pass runs, group its per-block assignments into
``s' = sec.rows * sec.cols`` "big blocks". For each of the ``s'`` positions
inside a big block, keep only the top ``k_per_bucket`` most frequent primary
motifs (iota_i). Any block whose motif is no longer allowed at its position is
reassigned to the nearest *allowed* primary motif and the weight tensor is
updated.

``k_per_bucket = round(K_primary * secondary.capacity)``.

If ``secondary.force_zero`` is true, the zero motif is always included in every
``iota_i``.
"""

import torch
import torch.nn.functional as F

from .momos2d import quantize_momos2D, blocks_to_tensor2D


_INTERNAL_KEYS = ("_motifs", "_nearest", "_layer_specs")


def quantize_hierarchical_momos2D(model, quant_cfg):
    primary_cfg = dict(quant_cfg["primary"])
    primary_cfg["method"] = "momos2d"
    stats = quantize_momos2D(model, primary_cfg) or {}

    Z = stats.pop("_motifs", None)
    M = stats.pop("_nearest", None)
    layer_specs = stats.pop("_layer_specs", None)

    if not quant_cfg.get("apply_secondary", False):
        return stats

    if Z is None or M is None or layer_specs is None:
        # Defensive — should never happen since we just ran primary.
        return stats

    secondary_cfg = dict(quant_cfg["secondary"])
    return _secondary_pass(
        Z=Z,
        M=M,
        layer_specs=layer_specs,
        rows=int(primary_cfg["rows"]),
        cols=int(primary_cfg["cols"]),
        sec_rows=int(secondary_cfg["rows"]),
        sec_cols=int(secondary_cfg["cols"]),
        capacity=float(secondary_cfg.get("capacity", 1.0)),
        force_zero=bool(secondary_cfg.get("force_zero", False)),
    )


def _per_param_grid(shape, rows, cols):
    if len(shape) < 2:
        h, w = 1, int(shape[0])
        batch = 1
    else:
        h, w = int(shape[-2]), int(shape[-1])
        batch = 1
        for d in shape[:-2]:
            batch *= int(d)
    R_p = (h + rows - 1) // rows
    C_p = (w + cols - 1) // cols
    return batch, R_p, C_p


def _secondary_pass(
    Z, M, layer_specs, rows, cols, sec_rows, sec_cols, capacity, force_zero
):
    s_prime = sec_rows * sec_cols
    K = int(Z.shape[0])
    device = Z.device

    with torch.no_grad():
        zero_idx = None
        if force_zero:
            zero_mask = Z.abs().sum(dim=1) == 0
            if zero_mask.any():
                zero_idx = int(zero_mask.nonzero(as_tuple=False)[0].item())

        # Per-position counts of primary motifs across all big blocks.
        counts = torch.zeros(s_prime, K, dtype=torch.long, device=device)
        grids = []
        offset = 0
        n_big_total = 0
        pos_idx_cache = torch.arange(s_prime, device=device)
        for param, n_blocks, _, shape in layer_specs:
            batch, R_p, C_p = _per_param_grid(shape, rows, cols)
            local = M[offset : offset + n_blocks].view(batch, R_p, C_p)
            offset += n_blocks

            pad_r = (-R_p) % sec_rows
            pad_c = (-C_p) % sec_cols
            if pad_r or pad_c:
                local = F.pad(local, (0, pad_c, 0, pad_r), value=K)
            R_big = local.shape[1] // sec_rows
            C_big = local.shape[2] // sec_cols
            grid = (
                local.view(batch, R_big, sec_rows, C_big, sec_cols)
                .permute(0, 1, 3, 2, 4)
                .contiguous()
            )

            flat = grid.view(-1, s_prime)
            combined = pos_idx_cache.expand_as(flat) * (K + 1) + flat
            c = torch.bincount(combined.flatten(), minlength=s_prime * (K + 1))
            counts += c.view(s_prime, K + 1)[:, :K]

            grids.append((param, shape, batch, R_p, C_p, R_big, C_big, grid))
            n_big_total += batch * R_big * C_big

        # k_per_bucket relative to big-block count: capacity=1 ⇒ k_per_bucket=n_big
        # (no mosaic compression, equivalent to standard MoMos); capped at K.
        k_per_bucket = max(1, min(K, int(round(n_big_total * capacity))))

        # iota_i per position.
        iotas = []
        for i in range(s_prime):
            c = counts[i].clone()
            chosen = []
            remaining = k_per_bucket
            if force_zero and zero_idx is not None:
                chosen.append(zero_idx)
                c[zero_idx] = -1
                remaining -= 1
            if remaining > 0:
                weights = c.clamp(min=0).float()
                nonzero = int((weights > 0).sum().item())
                n = min(remaining, nonzero)
                if n > 0:
                    sampled = torch.multinomial(weights, n, replacement=False).tolist()
                    chosen.extend(sampled)
            iotas.append(
                torch.tensor(sorted(set(chosen)), dtype=torch.long, device=device)
            )

        # Nearest allowed motif per position, precomputed as (s_prime, K+1).
        D = torch.cdist(Z, Z, p=2) ** 2
        mapping_stack = torch.empty(s_prime, K + 1, dtype=torch.long, device=device)
        for i, iota in enumerate(iotas):
            mapping_stack[i, :K] = iota[D[:, iota].argmin(dim=1)]
            mapping_stack[i, K] = K  # sentinel stays sentinel

        # Reassign per-param: vectorize across positions via gather.
        distortion = torch.zeros((), device=device)
        changed = torch.zeros((), dtype=torch.long, device=device)
        for param, shape, batch, R_p, C_p, R_big, C_big, grid in grids:
            flat_old = grid.view(-1, s_prime)  # (N_big, s_prime)
            new_flat = torch.gather(
                mapping_stack, 1, flat_old.t()
            ).t()  # (N_big, s_prime)

            new_grid = new_flat.view(batch, R_big, C_big, sec_rows, sec_cols)
            local_new = (
                new_grid.permute(0, 1, 3, 2, 4)
                .contiguous()
                .view(batch, R_big * sec_rows, C_big * sec_cols)
            )
            local_new = local_new[:, :R_p, :C_p].contiguous()
            new_blocks = Z[local_new.reshape(-1)]
            new_tensor = blocks_to_tensor2D(new_blocks, shape, rows, cols).reshape(
                param.shape
            )

            diff = new_tensor - param.data
            distortion += diff.square().sum()
            changed += (new_tensor != param.data).sum()
            param.data.copy_(new_tensor)

        return {
            "distortion": float(distortion.item()),
            "num_changed_weights": int(changed.item()),
        }
