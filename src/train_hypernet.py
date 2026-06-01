"""Training entry point for the LitMamba mask-predictor (hypernetwork).

Loads a quantized MLP checkpoint, builds per-(motif, layer) binary-mask
training samples via MotifMaskDataModule, and trains a LitMamba to predict
them. At the end of each training epoch, reconstructs the MLP from the
predicted masks and evaluates it on a held-out fold of the original image
dataset.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from coolname import generate_slug

from data import ImageDataModule, MotifMaskDataModule
from model.lit_mamba import LitMamba
from utils.init import resolve_runtime, setup_checkpoint_dir
from utils.callbacks import build_callbacks


@hydra.main(config_path="configs", config_name="hypernet", version_base="1.3")
def main(cfg: DictConfig):
    run_name = generate_slug()
    L.seed_everything(cfg.seed)

    accelerator, runtime_cfg = resolve_runtime(cfg.accelerator)

    # 1. Mask-training datamodule
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
    summary = mask_dm.summary()
    ds = mask_dm.train_dataset
    assert ds is not None

    # 2. Image datamodule for per-epoch validation on a fixed fold
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

    # 3. LitMamba with reconstruction + validation hooks
    input_dim = int(vd.in_channels) * int(vd.img_size) * int(vd.img_size)

    # Mamba2Config requires hidden_size * expand == num_heads * head_dim.
    # Derive head_dim so the constraint is satisfied automatically.
    expand = int(cfg.model.get("expand", 2))
    num_heads = int(cfg.model.num_heads)
    hidden_size = int(cfg.model.hidden_size)
    if (hidden_size * expand) % num_heads != 0:
        raise ValueError(
            f"hidden_size*expand ({hidden_size * expand}) must be divisible by num_heads ({num_heads})."
        )
    head_dim = (hidden_size * expand) // num_heads

    model = LitMamba(
        n_motifs=summary["n_motifs"],
        num_rows=summary["num_rows"],
        num_cols=summary["num_cols"],
        num_layers=summary["num_layers"],
        out_channels=summary["out_channels"],
        num_heads=num_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        state_size=cfg.model.state_size,
        num_hidden_layers=cfg.model.num_hidden_layers,
        n_groups=cfg.model.n_groups,
        hidden_act=cfg.model.hidden_act,
        time_step_min=cfg.model.time_step_min,
        time_step_max=cfg.model.time_step_max,
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
        epochs=cfg.epochs,
        motifs=ds.motifs,
        layer_shapes=ds.layer_shapes,
        motif_rows=cfg.motif_rows,
        motif_cols=cfg.motif_cols,
        rows=cfg.rows,
        cols=cfg.cols,
        n_rows_per_layer=ds.n_rows_per_layer,
        n_cols_per_layer=ds.n_cols_per_layer,
        image_datamodule=img_dm,
        mlp_input_dim=input_dim,
        mlp_num_classes=int(vd.num_classes),
        motif_batch_size=cfg.get("motif_batch_size", None),
        original_params=ds.original_params,
    )

    # 4. Callbacks + Trainer
    checkpoint_dir, unique_run_name, _ = setup_checkpoint_dir(
        log_dir=cfg.log_dir,
        dataset_name="hypernet",
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
            project=wandb_cfg.get("project", "momos-hypernet"),
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
        max_epochs=cfg.epochs,
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=50,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=True,
    )

    trainer.fit(model, datamodule=mask_dm)


if __name__ == "__main__":
    main()
