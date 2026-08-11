import unittest

import torch
import torch.nn as nn

from src.quantizers.fold_momos import quantize_fold_momos, _fold_kmeans_pass
from src.quantizers.momos2d import quantize_momos2D
from src.quantizers.momos_hierarchy import _sample_fold_ids


class BigLinearModel(nn.Module):
    """One sizeable linear layer, so fold sampling/k-means has enough blocks to
    work with (unlike the tiny fixed-value models in test_quant.py)."""

    def __init__(self, out_features=32, in_features=32, seed=0):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            self.fc.weight.copy_(torch.randn(out_features, in_features, generator=g))


def _base_cfg(apply_secondary):
    return {
        "primary": {"rows": 1, "cols": 2, "k": 16, "force_zero": True},
        "secondary": {
            "rows": 2,
            "cols": 2,
            "capacity": 0.5,
            "force_zero": True,
            "kmeans_n_init": 3,
        },
        "apply_secondary": apply_secondary,
    }


class TestSampleFoldIdsSharedHelper(unittest.TestCase):
    def test_deterministic_given_seed_and_always_includes_zero_when_forced(self):
        counts = torch.tensor([[5, 0, 3, 0], [0, 2, 0, 7]], dtype=torch.long)

        torch.manual_seed(0)
        ids_a = _sample_fold_ids(counts, k_per_bucket=2, force_zero=True, zero_idx=0)
        torch.manual_seed(0)
        ids_b = _sample_fold_ids(counts, k_per_bucket=2, force_zero=True, zero_idx=0)

        self.assertEqual(len(ids_a), 2)
        for a, b in zip(ids_a, ids_b):
            self.assertTrue(torch.equal(a, b))
        for iota in ids_a:
            self.assertIn(0, iota.tolist())
            self.assertLessEqual(len(iota), 2)

    def test_fold_with_no_occurrences_returns_only_forced_zero(self):
        counts = torch.tensor([[0, 0, 0, 0]], dtype=torch.long)
        ids = _sample_fold_ids(counts, k_per_bucket=3, force_zero=True, zero_idx=0)
        self.assertTrue(torch.equal(ids[0], torch.tensor([0], dtype=torch.long)))


class TestFoldMomos(unittest.TestCase):
    def test_without_apply_secondary_matches_plain_primary_pass(self):
        model_a = BigLinearModel(seed=1)
        model_b = BigLinearModel(seed=1)
        cfg = _base_cfg(apply_secondary=False)

        torch.manual_seed(123)
        stats_fold = quantize_fold_momos(model_a, cfg)

        primary_cfg = dict(_base_cfg(False)["primary"])
        primary_cfg["method"] = "momos2d"
        torch.manual_seed(123)
        stats_plain = quantize_momos2D(model_b, primary_cfg)

        self.assertTrue(torch.equal(model_a.fc.weight, model_b.fc.weight))
        self.assertAlmostEqual(
            stats_fold["distortion"], stats_plain["distortion"], places=4
        )
        self.assertEqual(
            stats_fold["num_changed_weights"], stats_plain["num_changed_weights"]
        )
        # Internal keys used to hand primary-pass state to the secondary stage
        # must not leak into the public stats dict.
        for key in ("_motifs", "_nearest", "_layer_specs", "_refined_motifs", "_refined_mosaic"):
            self.assertNotIn(key, stats_fold)

    def test_apply_secondary_reconstructs_from_shared_refined_codebook(self):
        torch.manual_seed(7)
        model = BigLinearModel(seed=2)
        cfg = _base_cfg(apply_secondary=True)

        stats = quantize_fold_momos(model, cfg)

        self.assertIn("num_motifs_used", stats)
        self.assertGreater(stats["num_motifs_used"], 0)
        self.assertLessEqual(stats["num_motifs_used"], cfg["primary"]["k"])
        self.assertGreaterEqual(stats["distortion"], 0.0)
        self.assertGreaterEqual(stats["num_changed_weights"], 0)
        for key in ("_motifs", "_nearest", "_layer_specs", "_refined_motifs", "_refined_mosaic"):
            self.assertNotIn(key, stats)

        # Round-trip: every distinct weight block in the model must be one of
        # the (<= num_motifs_used) reconstructed codebook values.
        blocks = model.fc.weight.detach().reshape(-1, 2)
        distinct = torch.unique(blocks, dim=0)
        self.assertLessEqual(distinct.shape[0], stats["num_motifs_used"])

    def test_force_zero_snaps_nearest_centroid_to_exact_zero(self):
        torch.manual_seed(11)
        model = BigLinearModel(seed=3)
        primary_cfg = dict(_base_cfg(True)["primary"])
        primary_cfg["method"] = "momos2d"
        primary_stats = quantize_momos2D(model, primary_cfg)

        fold_stats = _fold_kmeans_pass(
            Z=primary_stats["_motifs"],
            M=primary_stats["_nearest"],
            layer_specs=primary_stats["_layer_specs"],
            rows=1,
            cols=2,
            sec_rows=2,
            sec_cols=2,
            capacity=0.5,
            force_zero=True,
            n_init=3,
        )

        Z_refined = fold_stats["_refined_motifs"]
        self.assertAlmostEqual(float(Z_refined.norm(dim=1).min().item()), 0.0, places=5)

    def test_num_motifs_used_never_exceeds_primary_k(self):
        torch.manual_seed(5)
        model = BigLinearModel(seed=4)
        cfg = _base_cfg(True)
        cfg["secondary"]["capacity"] = 1.0  # sample every primary motif into every fold
        stats = quantize_fold_momos(model, cfg)
        self.assertLessEqual(stats["num_motifs_used"], cfg["primary"]["k"])


if __name__ == "__main__":
    unittest.main()
