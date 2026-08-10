"""End-to-end training of a Mamba hypernet that regresses MLP weights directly.

Unlike ``train_hypernet.py`` (which trains the hypernet on per-block binary
masks against a precomputed quantized MLP), this script wires the hypernet's
output as the MLP's parameters and trains it on the image classification loss.
There is no motif codebook, no discrete bottleneck — gradients flow from
``CrossEntropyLoss(MLP(x; θ_hyper), y)`` straight into the hypernet.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from coolname import generate_slug

from data import ImageDataModule
from model.lit_mamba_regressor import LitMambaRegressor
from utils.init import resolve_runtime, setup_checkpoint_dir
from utils.callbacks import build_callbacks


@hydra.main(config_path="configs", config_name="hypernet_regress", version_base="1.3")
def main(cfg: DictConfig):
    run_name = generate_slug()
    L.seed_everything(cfg.seed)

    _, runtime_cfg = resolve_runtime(cfg.accelerator)

    ds_cfg = cfg.dataset
    img_dm = ImageDataModule(
        dataset_name=ds_cfg.name,
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        img_size=ds_cfg.img_size,
        val_pct=ds_cfg.val_pct,
        test_pct=ds_cfg.test_pct,
        n_folds=ds_cfg.n_folds,
        fold=cfg.fold,
        kfold_seed=ds_cfg.kfold_seed,
        runtime=runtime_cfg,
    )
    img_dm.setup()

    input_dim = int(ds_cfg.in_channels) * int(ds_cfg.img_size) * int(ds_cfg.img_size)

    # Mamba2Config requires hidden_size * expand == num_heads * head_dim.
    expand = int(cfg.model.get("expand", 2))
    num_heads = int(cfg.model.num_heads)
    hidden_size = int(cfg.model.hidden_size)
    if (hidden_size * expand) % num_heads != 0:
        raise ValueError(
            f"hidden_size*expand ({hidden_size * expand}) must be divisible by num_heads ({num_heads})."
        )
    head_dim = (hidden_size * expand) // num_heads

    model = LitMambaRegressor(
        mlp_input_dim=input_dim,
        mlp_num_classes=int(ds_cfg.num_classes),
        rows=cfg.rows,
        cols=cfg.cols,
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
    )

    checkpoint_dir, unique_run_name, _ = setup_checkpoint_dir(
        log_dir=cfg.log_dir,
        dataset_name="hypernet_regress",
        prefix=cfg.prefix,
        run_name=run_name,
        model_name=cfg.model.name,
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
            project=wandb_cfg.get("project", "momos-hypernet-regress"),
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

    trainer.fit(model, datamodule=img_dm)


if __name__ == "__main__":
    main()
