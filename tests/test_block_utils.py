import unittest

import torch
import torch.nn as nn

import src.quantizers as quantizers
import src.quantizers.block_utils as block_utils


class MultiParamModel(nn.Module):
    """Model with multiple trainable parameters for iteration tests."""

    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.arange(6.0).view(2, 3), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(4), requires_grad=True)
        self.frozen = nn.Parameter(torch.zeros(2), requires_grad=False)


class EmptyParamModel(nn.Module):
    """Model with an empty trainable parameter."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor([]), requires_grad=True)


class VectorModel(nn.Module):
    """Single trainable parameter for simple tests."""

    def __init__(self, values):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(values, dtype=torch.float32))


class TestIterTrainableParams(unittest.TestCase):
    def test_yields_only_trainable_params(self):
        model = MultiParamModel()
        params = list(quantizers.iter_trainable_params(model))

        self.assertEqual(len(params), 2)
        self.assertTrue(params[0].requires_grad)
        self.assertTrue(params[1].requires_grad)

    def test_ignores_frozen_parameters(self):
        model = MultiParamModel()
        param_names = {p.shape for p in quantizers.iter_trainable_params(model)}

        # w1: (2, 3), w2: (4,)
        self.assertIn(torch.Size([2, 3]), param_names)
        self.assertIn(torch.Size([4]), param_names)
        # frozen: (2,) should not appear
        self.assertNotIn(torch.Size([2]), param_names)

    def test_ignores_empty_parameters(self):
        model = EmptyParamModel()
        params = list(quantizers.iter_trainable_params(model))

        self.assertEqual(len(params), 0)

    def test_order_preserved(self):
        model = MultiParamModel()
        params = list(quantizers.iter_trainable_params(model))

        # Should match model.parameters() order for trainable params
        expected = [p for p in model.parameters() if p.requires_grad and p.numel() > 0]
        self.assertEqual(len(params), len(expected))
        for actual, exp in zip(params, expected):
            self.assertTrue(torch.equal(actual, exp))


class TestTensorToBlocks(unittest.TestCase):
    def test_exact_division_no_padding(self):
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        blocks, n_params, shape = quantizers.tensor_to_blocks(tensor, block_size=2)

        self.assertEqual(blocks.shape, (2, 2))
        self.assertEqual(n_params, 4)
        self.assertEqual(shape, torch.Size([4]))
        self.assertTrue(
            torch.equal(
                blocks, torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
            )
        )

    def test_padding_with_zeros(self):
        tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        blocks, n_params, shape = quantizers.tensor_to_blocks(tensor, block_size=2)

        self.assertEqual(blocks.shape, (2, 2))
        self.assertEqual(n_params, 3)
        self.assertTrue(
            torch.equal(
                blocks, torch.tensor([[1.0, 2.0], [3.0, 0.0]], dtype=torch.float32)
            )
        )

    def test_block_size_equals_tensor_size(self):
        tensor = torch.tensor([5.0, 6.0, 7.0], dtype=torch.float32)
        blocks, n_params, shape = quantizers.tensor_to_blocks(tensor, block_size=3)

        self.assertEqual(blocks.shape, (1, 3))
        self.assertEqual(n_params, 3)
        self.assertTrue(torch.equal(blocks.flatten(), tensor))

    def test_multidimensional_tensor(self):
        tensor = torch.arange(12.0).view(3, 4)
        blocks, n_params, shape = quantizers.tensor_to_blocks(tensor, block_size=4)

        self.assertEqual(blocks.shape, (3, 4))
        self.assertEqual(n_params, 12)
        self.assertEqual(shape, torch.Size([3, 4]))

    def test_invalid_block_size_zero(self):
        tensor = torch.tensor([1.0], dtype=torch.float32)
        with self.assertRaises(ValueError):
            quantizers.tensor_to_blocks(tensor, block_size=0)

    def test_invalid_block_size_negative(self):
        tensor = torch.tensor([1.0], dtype=torch.float32)
        with self.assertRaises(ValueError):
            quantizers.tensor_to_blocks(tensor, block_size=-1)

    def test_single_element(self):
        tensor = torch.tensor([42.0], dtype=torch.float32)
        blocks, n_params, shape = quantizers.tensor_to_blocks(tensor, block_size=1)

        self.assertEqual(blocks.shape, (1, 1))
        self.assertEqual(n_params, 1)
        self.assertEqual(blocks[0, 0].item(), 42.0)

    def test_preserves_dtype(self):
        tensor = torch.tensor([1.0, 2.0], dtype=torch.float64)
        blocks, _, _ = quantizers.tensor_to_blocks(tensor, block_size=2)

        self.assertEqual(blocks.dtype, torch.float64)

    def test_preserves_device(self):
        tensor = torch.tensor([1.0, 2.0], dtype=torch.float32)
        blocks, _, _ = quantizers.tensor_to_blocks(tensor, block_size=2)

        self.assertEqual(blocks.device, tensor.device)


class TestBlocksToTensor(unittest.TestCase):
    def test_reconstruct_exact_tensor(self):
        original = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        blocks, n_params, shape = quantizers.tensor_to_blocks(original, block_size=2)
        rebuilt = quantizers.blocks_to_tensor(blocks, n_params=n_params, shape=shape)

        self.assertTrue(torch.equal(rebuilt, original))

    def test_reconstruct_padded_tensor(self):
        original = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        blocks, n_params, shape = quantizers.tensor_to_blocks(original, block_size=2)
        rebuilt = quantizers.blocks_to_tensor(blocks, n_params=n_params, shape=shape)

        self.assertTrue(torch.equal(rebuilt, original))
        self.assertEqual(rebuilt.shape, original.shape)

    def test_multidimensional_reconstruction(self):
        original = torch.arange(6.0).view(2, 3)
        blocks, n_params, shape = quantizers.tensor_to_blocks(original, block_size=3)
        rebuilt = quantizers.blocks_to_tensor(blocks, n_params=n_params, shape=shape)

        self.assertTrue(torch.equal(rebuilt, original))
        self.assertEqual(rebuilt.shape, torch.Size([2, 3]))

    def test_roundtrip_large_tensor(self):
        original = torch.randn(100)
        blocks, n_params, shape = quantizers.tensor_to_blocks(original, block_size=7)
        rebuilt = quantizers.blocks_to_tensor(blocks, n_params=n_params, shape=shape)

        self.assertTrue(torch.allclose(rebuilt, original))
        self.assertEqual(rebuilt.shape, original.shape)


class TestCountTotalBlocks(unittest.TestCase):
    def test_counts_all_trainable_parameters(self):
        model = MultiParamModel()
        # w1: 6 elements, w2: 4 elements, block_size=2
        # w1: ceil(6/2) = 3, w2: ceil(4/2) = 2
        total = quantizers.count_total_blocks(model, block_size=2)

        self.assertEqual(total, 5)

    def test_ignores_frozen_parameters(self):
        model = MultiParamModel()
        # frozen has 2 elements but requires_grad=False
        with_frozen = quantizers.count_total_blocks(model, block_size=2)

        # Manually count only trainable
        expected = sum(
            (p.numel() + 1) // 2
            for p in model.parameters()
            if p.requires_grad and p.numel() > 0
        )

        self.assertEqual(with_frozen, expected)

    def test_block_size_larger_than_tensor(self):
        class Simple(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(
                    torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
                )

        model = Simple()
        total = quantizers.count_total_blocks(model, block_size=10)
        # Should be 1 block even if block_size > numel
        self.assertEqual(total, 1)

    def test_empty_model(self):
        model = EmptyParamModel()
        total = quantizers.count_total_blocks(model, block_size=4)

        self.assertEqual(total, 0)

    def test_invalid_block_size(self):
        model = MultiParamModel()
        with self.assertRaises(ValueError):
            quantizers.count_total_blocks(model, block_size=0)

    def test_returns_integer(self):
        model = MultiParamModel()
        total = quantizers.count_total_blocks(model, block_size=3)

        self.assertIsInstance(total, int)


class TestKFromCapacity(unittest.TestCase):
    def test_basic_capacity_conversion(self):
        # 10 elements with block_size=1 => 10 blocks
        # capacity=0.5 => k=5
        model = VectorModel(values=list(range(10)))
        k = quantizers.k_from_capacity(model, block_size=1, capacity=0.5)

        self.assertEqual(k, 5)

    def test_capacity_clamped_to_minimum_one(self):
        model = VectorModel(values=[1.0, 2.0, 3.0])
        k = quantizers.k_from_capacity(model, block_size=1, capacity=0.01)

        self.assertGreaterEqual(k, 1)

    def test_capacity_clamped_to_maximum_total_blocks(self):
        model = VectorModel(values=[1.0, 2.0, 3.0])
        k = quantizers.k_from_capacity(model, block_size=1, capacity=2.0)

        # Should not exceed total number of blocks
        total_blocks = quantizers.count_total_blocks(model, block_size=1)
        self.assertLessEqual(k, total_blocks)
        self.assertEqual(k, 3)  # ceil(3/1) = 3 blocks

    def test_invalid_capacity_zero(self):
        model = VectorModel(values=[1.0])
        with self.assertRaises(ValueError):
            quantizers.k_from_capacity(model, block_size=1, capacity=0)

    def test_invalid_capacity_negative(self):
        model = VectorModel(values=[1.0])
        with self.assertRaises(ValueError):
            quantizers.k_from_capacity(model, block_size=1, capacity=-0.5)

    def test_small_model_with_large_capacity(self):
        model = VectorModel(values=[1.0, 2.0])
        k = quantizers.k_from_capacity(model, block_size=2, capacity=0.99)

        # 1 block total, should clamp to 1
        self.assertEqual(k, 1)


class TestResolveChunkSizeBlocks(unittest.TestCase):
    def test_default_chunk_size_resolved(self):
        # Should return a reasonable chunk size based on default budget
        chunk = block_utils._resolve_chunk_size_blocks(n_blocks=100, n_motifs=10)

        self.assertGreaterEqual(chunk, 1)
        self.assertLessEqual(chunk, 100)

    def test_custom_chunk_size_budget(self):
        # Small budget should result in smaller chunk
        chunk_small = block_utils._resolve_chunk_size_blocks(
            n_blocks=100,
            n_motifs=10,
            chunk_size=1.0,  # 1 MB
        )
        chunk_large = block_utils._resolve_chunk_size_blocks(
            n_blocks=100,
            n_motifs=10,
            chunk_size=100.0,  # 100 MB
        )

        self.assertLessEqual(chunk_small, chunk_large)

    def test_chunk_size_cannot_exceed_n_blocks(self):
        chunk = block_utils._resolve_chunk_size_blocks(
            n_blocks=5, n_motifs=2, chunk_size=999999.0
        )

        self.assertEqual(chunk, 5)

    def test_minimum_chunk_size_is_one(self):
        chunk = block_utils._resolve_chunk_size_blocks(
            n_blocks=10, n_motifs=1000000, chunk_size=1e-6
        )

        self.assertGreaterEqual(chunk, 1)

    def test_invalid_chunk_size(self):
        with self.assertRaises(ValueError):
            block_utils._resolve_chunk_size_blocks(
                n_blocks=10, n_motifs=5, chunk_size=0
            )

    def test_zero_n_blocks_returns_one(self):
        chunk = block_utils._resolve_chunk_size_blocks(n_blocks=0, n_motifs=5)

        self.assertEqual(chunk, 1)

    def test_dtype_affects_bytes_per_element(self):
        chunk_f32 = block_utils._resolve_chunk_size_blocks(
            n_blocks=100, n_motifs=10, chunk_size=1.0, dtype=torch.float32
        )
        chunk_f64 = block_utils._resolve_chunk_size_blocks(
            n_blocks=100, n_motifs=10, chunk_size=1.0, dtype=torch.float64
        )

        # f64 uses 2x bytes, should result in smaller chunk
        self.assertLessEqual(chunk_f64, chunk_f32)


class TestResolveProgressEveryElements(unittest.TestCase):
    def test_explicit_value_returned(self):
        result = block_utils._resolve_progress_every_elements(
            total_elements=1000, progress_every_elements=50
        )

        self.assertEqual(result, 50)

    def test_default_is_twentieth_of_total(self):
        result = block_utils._resolve_progress_every_elements(total_elements=1000)

        self.assertEqual(result, 50)  # 1000 // 20

    def test_minimum_value_is_one(self):
        result = block_utils._resolve_progress_every_elements(total_elements=5)

        self.assertGreaterEqual(result, 1)

    def test_invalid_explicit_value_raises_error(self):
        with self.assertRaises(ValueError):
            block_utils._resolve_progress_every_elements(
                total_elements=100, progress_every_elements=0
            )

    def test_negative_explicit_value_raises_error(self):
        with self.assertRaises(ValueError):
            block_utils._resolve_progress_every_elements(
                total_elements=100, progress_every_elements=-10
            )

    def test_zero_total_elements_returns_one(self):
        result = block_utils._resolve_progress_every_elements(total_elements=0)

        self.assertEqual(result, 1)


class TestBuildSwapMotif(unittest.TestCase):
    def _skewed_idx(self):
        """4 motifs: 0→50, 1→25, 2→20, 3→5. Sorted by freq asc: [3, 2, 1, 0]."""
        return torch.cat(
            [
                torch.zeros(50, dtype=torch.long),
                torch.ones(25, dtype=torch.long),
                torch.full((20,), 2, dtype=torch.long),
                torch.full((5,), 3, dtype=torch.long),
            ]
        )

    def test_returns_callable(self):
        swap = block_utils.build_swap_motif([0.0], [1.0], probability=1.0)
        self.assertTrue(callable(swap))

    def test_probability_zero_no_change(self):
        idx = self._skewed_idx()
        swap = block_utils.build_swap_motif([0.0], [1.0], probability=0.0, window=0.1)
        new_idx = swap(idx)
        self.assertTrue(torch.equal(new_idx, idx))

    def test_probability_one_source_fully_replaced(self):
        # rank asc: [3,2,1,0]; from perc 0.0 → motif 3; to perc 1.0 → motif 0
        idx = self._skewed_idx()
        swap = block_utils.build_swap_motif([0.0], [1.0], probability=1.0, window=0.1)
        new_idx = swap(idx)
        self.assertEqual((new_idx == 3).sum().item(), 0)

    def test_probability_one_target_absorbs_source_count(self):
        idx = self._skewed_idx()
        swap = block_utils.build_swap_motif([0.0], [1.0], probability=1.0, window=0.1)
        new_idx = swap(idx)
        # motif 0 had 50; gains the 5 from motif 3
        self.assertEqual((new_idx == 0).sum().item(), 55)

    def test_output_shape_unchanged(self):
        idx = self._skewed_idx()
        swap = block_utils.build_swap_motif([0.0], [1.0], probability=1.0, window=0.1)
        self.assertEqual(swap(idx).shape, idx.shape)

    def test_input_tensor_not_mutated(self):
        idx = self._skewed_idx()
        original = idx.clone()
        block_utils.build_swap_motif([0.0], [1.0], probability=1.0, window=0.1)(idx)
        self.assertTrue(torch.equal(idx, original))

    def test_output_contains_only_valid_motif_ids(self):
        idx = self._skewed_idx()
        valid_ids = set(idx.unique().tolist())
        swap = block_utils.build_swap_motif([0.0], [1.0], probability=1.0, window=0.1)
        self.assertTrue(set(swap(idx).unique().tolist()).issubset(valid_ids))

    def test_unaffected_motifs_count_unchanged(self):
        # Only motif 3 (bottom percentile) is in the source window; 1 and 2 stay intact
        idx = self._skewed_idx()
        swap = block_utils.build_swap_motif([0.0], [1.0], probability=1.0, window=0.1)
        new_idx = swap(idx)
        self.assertEqual((new_idx == 1).sum().item(), 25)
        self.assertEqual((new_idx == 2).sum().item(), 20)

    def test_partial_probability_reduces_source_count(self):
        torch.manual_seed(0)
        idx = self._skewed_idx()
        original_count = (idx == 3).sum().item()
        swap = block_utils.build_swap_motif([0.0], [1.0], probability=0.5, window=0.1)
        new_idx = swap(idx)
        self.assertLess((new_idx == 3).sum().item(), original_count)

    def test_single_unique_motif_is_noop(self):
        # Source and target are the same motif — output must equal input
        idx = torch.zeros(20, dtype=torch.long)
        swap = block_utils.build_swap_motif([0.0], [1.0], probability=1.0, window=0.5)
        self.assertTrue(torch.equal(swap(idx), idx))

    def test_multiple_swap_pairs_both_applied(self):
        # pair 1 window hits motif 3 (rank 0); pair 2 window hits motifs 3 and 2 (ranks 0-1)
        # With prob=1 both low-freq motifs should disappear
        idx = self._skewed_idx()
        swap = block_utils.build_swap_motif(
            from_percentiles=[0.0, 0.25],
            to_percentiles=[1.0, 1.0],
            probability=1.0,
            window=0.1,
        )
        new_idx = swap(idx)
        self.assertEqual((new_idx == 3).sum().item(), 0)
        self.assertEqual((new_idx == 2).sum().item(), 0)

    def test_swap_is_deterministic_with_fixed_seed(self):
        idx = self._skewed_idx()
        swap = block_utils.build_swap_motif([0.0], [1.0], probability=0.7, window=0.1)
        torch.manual_seed(99)
        result_a = swap(idx).clone()
        torch.manual_seed(99)
        result_b = swap(idx).clone()
        self.assertTrue(torch.equal(result_a, result_b))


if __name__ == "__main__":
    unittest.main()
