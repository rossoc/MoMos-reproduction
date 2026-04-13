"""Training entry point using PyTorch Lightning and Hydra."""

import os

import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import WandbLogger
from coolname import generate_slug

from data import ImageDataModule
from model.lit_module import LitMLP


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    """Run training with PyTorch Lightning and Hydra config management."""
    # Set seed for reproducibility
    L.seed_everything(cfg.seed)

    # Resolve num_workers from runtime profile
    num_workers = cfg.num_workers
    if num_workers is None:
        if cfg.accelerator == "cuda":
            num_workers = 4
        elif cfg.accelerator == "cpu":
            num_workers = 4
        else:
            num_workers = 0

    # Create DataModule
    datamodule: ImageDataModule = ImageDataModule(
        dataset_name=cfg.dataset.name,
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        img_size=cfg.dataset.img_size,
        val_pct=cfg.dataset.val_pct,
        test_pct=cfg.dataset.test_pct,
        runtime={
            "num_workers": num_workers,
            "pin_memory": cfg.accelerator == "cuda",
            "persistent_workers": num_workers > 0,
            "prefetch_factor": 2 if num_workers > 0 else None,
        },
    )

    # Calculate input dimension
    input_dim = cfg.dataset.in_channels * cfg.dataset.img_size * cfg.dataset.img_size

    # Create LightningModule
    model = LitMLP(
        input_dim=input_dim,
        num_classes=cfg.dataset.num_classes,
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
        epochs=cfg.epochs,
    )

    # Setup callbacks
    callbacks = []

    # Learning rate monitor
    callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    # Model checkpointing (uses Hydra's output dir)
    checkpoint_dir = os.path.join(cfg.log_dir, cfg.prefix or f"{cfg.dataset.name}_mlp")
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=False,
    )
    callbacks.append(checkpoint_callback)

    # Early stopping (optional)
    if cfg.patience is not None and cfg.patience > 0:
        early_stopping = EarlyStopping(
            monitor="val/loss",
            mode="min",
            patience=cfg.patience,
            verbose=True,
        )
        callbacks.append(early_stopping)

    # Setup W&B logger (if enabled)
    # When disabled, use False to prevent Lightning from creating default lightning_logs/ directory
    if cfg.get("wandb", {}).get("enabled", False):
        wandb_cfg = cfg.wandb
        run_name = wandb_cfg.get("name") or generate_slug()
        logger = WandbLogger(
            project=wandb_cfg.get("project", "momos-reproduction"),
            entity=wandb_cfg.get("entity", None),
            name=run_name,
            tags=wandb_cfg.get("tags", []),
            log_model=wandb_cfg.get("log_model", False),
            save_dir=cfg.log_dir,
        )
        # Log full Hydra config to W&B
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
        logger.experiment.config.update(
            {
                "input_dim": input_dim,
                "num_workers": num_workers,
            }
        )
    else:
        # Explicitly disable logging to prevent lightning_logs/ directory creation
        logger = False

    # Create Trainer
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

    # Save initial checkpoint before training
    init_ckpt_path = os.path.join(checkpoint_dir, "init.ckpt")
    trainer.save_checkpoint(init_ckpt_path)
    print(f"Saved initial checkpoint: {init_ckpt_path}")

    # Train
    trainer.fit(model, datamodule=datamodule)

    # Test with best checkpoint
    if checkpoint_callback.best_model_path:
        print(f"\nLoading best checkpoint: {checkpoint_callback.best_model_path}")
        print(f"Best validation accuracy: {checkpoint_callback.best_model_score:.4f}")

    # Final test evaluation
    print("\nEvaluating on test set...")
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
