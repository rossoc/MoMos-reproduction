"""Oracle sanity check for the hypernet pipeline.

Bypasses the Mamba mask predictor and feeds the ground-truth per-layer motif
grids (i.e. what a 100%-accurate model would argmax to) into the same
reconstruction + validation path used by ``LitMamba.build_mlp`` and
``LitMamba.on_train_epoch_end``. Prints, side-by-side, the source-checkpoint
MLP accuracy and the oracle-reconstructed MLP accuracy on the same validation
fold. Because ``MotifMaskDataset.motifs = all_blocks.unique(dim=0)``, the two
should agree to floating-point noise; any gap indicates a bug in the
reconstruction or evaluation glue rather than in the Mamba model.
"""

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig
import lightning as L

from data import ImageDataModule, MotifMaskDataModule
from model.lit_module import LitMLP
from quantizers.momos2d import blocks_to_tensor2D
from quantizers.block_utils import iter_trainable_params
from view.weight_distribution import load_model
from utils.init import resolve_runtime


def _evaluate(model: nn.Module, val_loader, device: torch.device) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0.0
    n = 0
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            bs = x.size(0)
            total_loss += loss.item() * bs
            total_correct += logits.argmax(dim=1).eq(y).float().sum().item()
            n += bs
    return total_correct / n, total_loss / n


def _oracle_build_mlp(
    ds,
    motif_rows: int,
    motif_cols: int,
    mlp_input_dim: int,
    mlp_num_classes: int,
    device: torch.device,
) -> tuple[LitMLP, float]:
    """Mirror of ``LitMamba.build_mlp`` that uses ``ds.layer_grids`` directly."""
    lit_mlp = LitMLP(input_dim=mlp_input_dim, num_classes=mlp_num_classes).to(device)
    motifs = ds.motifs.to(device)

    sse_total = 0.0
    n_total = 0
    with torch.no_grad():
        params = list(iter_trainable_params(lit_mlp.model))
        for L_idx, param in enumerate(params):
            shape = ds.layer_shapes[L_idx]
            orig_h, orig_w = int(shape[-2]), int(shape[-1])
            num_blocks_h = (orig_h + motif_rows - 1) // motif_rows
            num_blocks_w = (orig_w + motif_cols - 1) // motif_cols

            predicted_M = ds.layer_grids[L_idx][:num_blocks_h, :num_blocks_w].to(device)

            blocks = motifs[predicted_M].reshape(
                num_blocks_h * num_blocks_w, motif_rows * motif_cols
            )
            reconstructed = blocks_to_tensor2D(blocks, shape, motif_rows, motif_cols)
            recon = reconstructed.view_as(param).to(param.dtype)
            param.data.copy_(recon)

            orig = ds.original_params[L_idx].to(device=device, dtype=recon.dtype)
            sse_total += float(((recon - orig) ** 2).sum().item())
            n_total += orig.numel()

    return lit_mlp, sse_total / n_total


@hydra.main(config_path="configs", config_name="hypernet", version_base="1.3")
def main(cfg: DictConfig):
    L.seed_everything(cfg.seed)
    _, runtime_cfg = resolve_runtime(cfg.accelerator)

    mask_dm = MotifMaskDataModule(
        checkpoint_path=cfg.checkpoint_path,
        rows=cfg.rows,
        cols=cfg.cols,
        motif_rows=cfg.motif_rows,
        motif_cols=cfg.motif_cols,
        batch_size=cfg.batch_size,
        runtime=runtime_cfg,
        motif_batch_size=cfg.get("motif_batch_size", None),
    )
    mask_dm.setup()
    ds = mask_dm.train_dataset
    assert ds is not None

    vd = cfg.validation_dataset
    img_dm = ImageDataModule(
        dataset_name=vd.name,
        data_dir=cfg.data_dir,
        batch_size=128,
        img_size=vd.img_size,
        val_pct=vd.val_pct,
        test_pct=vd.test_pct,
        n_folds=vd.n_folds,
        fold=vd.fold,
        kfold_seed=vd.kfold_seed,
        runtime=runtime_cfg,
    )
    img_dm.setup()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = int(vd.in_channels) * int(vd.img_size) * int(vd.img_size)

    lit_mlp, oracle_mse = _oracle_build_mlp(
        ds=ds,
        motif_rows=int(cfg.motif_rows),
        motif_cols=int(cfg.motif_cols),
        mlp_input_dim=input_dim,
        mlp_num_classes=int(vd.num_classes),
        device=device,
    )
    oracle_acc, oracle_loss = _evaluate(lit_mlp, img_dm.val_dataloader(), device)

    raw_mlp = load_model(cfg.checkpoint_path).to(device)
    raw_acc, raw_loss = _evaluate(raw_mlp, img_dm.val_dataloader(), device)

    print(f"n_motifs={ds.n_motifs}  n_layers={ds.n_layers}")
    print(f"checkpoint    val_acc={raw_acc:.4f}    val_loss={raw_loss:.4f}")
    print(
        f"oracle recon  val_acc={oracle_acc:.4f}    val_loss={oracle_loss:.4f}    "
        f"recon_mse={oracle_mse:.3e}"
    )


if __name__ == "__main__":
    main()
