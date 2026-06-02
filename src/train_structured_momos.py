"""Training entry point for `LitStructuredMomos`.

The hypernet is no longer trained to *reproduce* the original mask grid.
Instead, it learns to *propose* a fresh MLP: per step, Mamba emits motif logits
at every spatial position, Gumbel-softmax (STE) makes the selection
differentiable, motifs are tiled into MLP weights, and classification CE on
the task batch backprops through the whole chain. Mask sparsity / one-hot
behaviour is encouraged by an annealed entropy penalty plus a load-balance KL
(see `model.lit_structured_momos.LitStructuredMomos`).

Only the *motif library* and *per-layer geometry* come from the source
quantized checkpoint (read via `MotifMaskDataset`); training data is the
image task data via `ImageDataModule`.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from coolname import generate_slug

from data import ImageDataModule
from data.momos_mask_datamodule import MotifMaskDataset
from model.lit_structured_momos import LitStructuredMomos
from utils.init import resolve_runtime, setup_checkpoint_dir
from utils.callbacks import build_callbacks


def run_training(cfg: DictConfig) -> float:
    """Run one training; return best ``val/acc``."""
    run_name = generate_slug()
    L.seed_everything(cfg.seed)

    accelerator, runtime_cfg = resolve_runtime(cfg.accelerator)

    # 1. Extract motif library + per-layer geometry from the source checkpoint.
    #    We use the dataset directly (not the DataModule) since we don't need
    #    its sampler — training data comes from the task ImageDataModule.
    motif_ds = MotifMaskDataset(
        checkpoint_path=cfg.checkpoint_path,
        rows=cfg.rows,
        cols=cfg.cols,
        motif_rows=cfg.motif_rows,
        motif_cols=cfg.motif_cols,
        mode="per_layer",
    )

    # 2. Task image data (the real training signal).
    td = cfg.task_dataset
    img_dm = ImageDataModule(
        dataset_name=td.name,
        data_dir=cfg.data_dir,
        batch_size=int(td.batch_size),
        img_size=int(td.img_size),
        val_pct=td.val_pct,
        test_pct=td.test_pct,
        n_folds=td.n_folds,
        fold=td.fold,
        kfold_seed=td.kfold_seed,
        runtime=runtime_cfg,
    )
    img_dm.setup()

    input_dim = int(td.in_channels) * int(td.img_size) * int(td.img_size)

    # 3. Mamba head_dim is derived so hidden_size*expand == num_heads*head_dim.
    expand = int(cfg.model.get("expand", 2))
    num_heads = int(cfg.model.num_heads)
    hidden_size = int(cfg.model.hidden_size)
    if (hidden_size * expand) % num_heads != 0:
        raise ValueError(
            f"hidden_size*expand ({hidden_size * expand}) must be divisible by "
            f"num_heads ({num_heads})."
        )
    head_dim = (hidden_size * expand) // num_heads

    # Mamba's per-layer grid covers the largest layer; smaller layers crop in.
    num_rows = max(motif_ds.n_rows_per_layer)
    num_cols = max(motif_ds.n_cols_per_layer)
    out_channels = int(cfg.rows) * int(cfg.cols)

    # 4. LitModule.
    model = LitStructuredMomos(
        # Mamba config
        n_motifs=motif_ds.n_motifs,
        num_heads=num_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        num_rows=num_rows,
        num_cols=num_cols,
        num_layers=motif_ds.n_layers,
        state_size=int(cfg.model.state_size),
        num_hidden_layers=int(cfg.model.num_hidden_layers),
        n_groups=int(cfg.model.n_groups),
        hidden_act=str(cfg.model.hidden_act),
        time_step_min=float(cfg.model.time_step_min),
        time_step_max=float(cfg.model.time_step_max),
        out_channels=out_channels,
        output_layers=int(cfg.get("output_layers", 2)),
        # Motif library + layer geometry
        motifs=motif_ds.motifs,
        layer_shapes=motif_ds.layer_shapes,
        motif_rows=int(cfg.motif_rows),
        motif_cols=int(cfg.motif_cols),
        rows=int(cfg.rows),
        cols=int(cfg.cols),
        n_rows_per_layer=motif_ds.n_rows_per_layer,
        n_cols_per_layer=motif_ds.n_cols_per_layer,
        # Target MLP geometry
        mlp_input_dim=input_dim,
        mlp_num_classes=int(td.num_classes),
        # Optimization
        learning_rate=float(cfg.model.learning_rate),
        weight_decay=float(cfg.model.weight_decay),
        epochs=int(cfg.epochs),
        # Memory
        motif_chunk_size=cfg.get("motif_chunk_size", None),
        mamba_chunk_size=cfg.get("mamba_chunk_size", None),
    )

    # 5. Checkpoints + callbacks + (optional) WandB.
    checkpoint_dir, unique_run_name, _ = setup_checkpoint_dir(
        log_dir=cfg.log_dir,
        dataset_name="structured_momos",
        prefix=cfg.prefix,
        run_name=run_name,
    )

    has_logger = cfg.get("wandb", {}).get("enabled", False)
    callbacks = build_callbacks(
        cfg=cfg,
        checkpoint_dir=checkpoint_dir,
        unique_run_name=unique_run_name,
        has_logger=has_logger,
    )

    if has_logger:
        wandb_cfg = cfg.wandb
        wandb_run_name = wandb_cfg.get("name") or unique_run_name
        logger = WandbLogger(
            project=wandb_cfg.get("project", "momos-structured"),
            entity=wandb_cfg.get("entity", None),
            name=wandb_run_name,
            tags=wandb_cfg.get("tags", []),
            log_model=wandb_cfg.get("log_model", False),
            save_dir=cfg.log_dir,
        )
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))  # type: ignore
    else:
        logger = False

    trainer = L.Trainer(
        max_epochs=int(cfg.epochs),
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=50,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=True,
        gradient_clip_val=float(cfg.get("grad_clip", 0.0)) or None,
        # Sanity-check val runs *before* training and uses the same Mamba
        # forward path as the real val step. Skip it — the end-of-epoch val
        # will catch real bugs.
        num_sanity_val_steps=0,
    )

    trainer.fit(model, datamodule=img_dm)
    return float(model.best_val_acc)


@hydra.main(
    config_path="configs", config_name="structured_momos", version_base="1.3"
)
def main(cfg: DictConfig):
    run_training(cfg)


if __name__ == "__main__":
    main()
