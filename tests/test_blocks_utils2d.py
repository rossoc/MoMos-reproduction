import pytest
import torch
from src.quantizers.momos2d import blocks_to_tensor2D, tensor2D_to_blocks


def test_round_trip_basic():
    """Test standard case with perfect divisibility."""
    x = torch.randn(1, 4, 4)
    rows, cols = 2, 2
    blocks, _, shape = tensor2D_to_blocks(x, rows, cols)
    reconstructed = blocks_to_tensor2D(blocks, shape, rows, cols)
    assert torch.equal(x, reconstructed)


def test_padding_logic():
    """Test dimensions that require padding (e.g., 5x5 into 2x2 blocks)."""
    x = torch.arange(25).view(1, 5, 5).float()
    rows, cols = 2, 2
    blocks, _, shape = tensor2D_to_blocks(x, rows, cols)

    # Expected: 5x5 padded to 6x6 -> 9 blocks of size 4
    # Calculation: (6/2) * (6/2) = 3 * 3 = 9 blocks
    assert blocks.shape == (9, 4)

    reconstructed = blocks_to_tensor2D(blocks, shape, rows, cols)
    assert torch.equal(x, reconstructed)
    assert reconstructed.shape == (1, 5, 5)


def test_high_dimensional_batch():
    """Test multiple leading dimensions (e.g., [2, 3, 8, 8])."""
    shape = (2, 3, 8, 8)
    x = torch.randn(*shape)
    rows, cols = 4, 4
    blocks, _, orig_shape = tensor2D_to_blocks(x, rows, cols)
    reconstructed = blocks_to_tensor2D(blocks, orig_shape, rows, cols)

    assert reconstructed.shape == shape
    assert torch.equal(x, reconstructed)


def test_invalid_block_size():
    """Ensure non-positive block sizes raise ValueError."""
    x = torch.randn(1, 4, 4)
    with pytest.raises(ValueError):
        tensor2D_to_blocks(x, 0, 2)
    with pytest.raises(ValueError):
        tensor2D_to_blocks(x, 2, -1)


def test_single_row_col_blocks():
    """Edge case: block size is 1x1."""
    x = torch.randn(1, 3, 3)
    blocks, _, shape = tensor2D_to_blocks(x, 1, 1)
    assert blocks.shape == (9, 1)
    assert torch.equal(x, blocks_to_tensor2D(blocks, shape, 1, 1))


@pytest.mark.parametrize(
    "h, w, r, c",
    [
        (10, 10, 3, 3),
        (7, 13, 5, 2),
        (1, 1, 10, 10),
    ],
)
def test_random_shapes(h, w, r, c):
    """Stress test various combinations of shapes and block sizes."""
    x = torch.randn(2, h, w)
    blocks, _, shape = tensor2D_to_blocks(x, r, c)
    reconstructed = blocks_to_tensor2D(blocks, shape, r, c)
    assert torch.equal(x, reconstructed)
