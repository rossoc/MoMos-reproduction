"""Training entry point: fit an MLP to memorize a dictionary of weight blocks."""

import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from coolname import generate_slug

from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from data.blocks_datamodule import BlocksDataModule
from data import ImageDataModule
from model.lit_index_to_block import LitIndexToBlock
from model.lit_module import LitMLP
from utils.init import resolve_runtime, setup_checkpoint_dir
from utils.io_model import load_model_from_checkpoint
from utils.fisher import compute_fisher_weights

import torch
import torch.nn as nn


def _maybe_compute_fisher(cfg, runtime_cfg):
    """Compute empirical-Fisher importance weights, or return None if disabled."""
    fcfg = cfg.dataset.get("fisher", None)
    if not fcfg or not fcfg.get("enabled", False):
        return None

    img_dm = ImageDataModule(
        dataset_name=fcfg.image_dataset,
        batch_size=int(fcfg.batch_size),
        img_size=int(fcfg.img_size),
        val_pct=fcfg.get("val_pct", 0.05),
        test_pct=fcfg.get("test_pct", 1.0),
        n_folds=int(fcfg.get("n_folds", 5)),
        kfold_seed=int(fcfg.get("kfold_seed", 10)),
        runtime=runtime_cfg,
        data_dir=cfg.data_dir,
    )
    img_dm.setup()

    in_channels = int(fcfg.in_channels)
    img_size = int(fcfg.img_size)
    num_classes = int(fcfg.num_classes)
    input_dim = in_channels * img_size * img_size

    src_lit = LitMLP(input_dim=input_dim, num_classes=num_classes)
    loaded = load_model_from_checkpoint(cfg.dataset.source_checkpoint)
    if loaded is None:
        raise FileNotFoundError(cfg.dataset.source_checkpoint)
    sd = loaded["state_dict"]
    if isinstance(sd, nn.Module):
        sd = sd.state_dict()
    # Accept either Lightning-style ("model.xxx") or raw inner state_dicts.
    if any(k.startswith("model.") for k in sd):
        src_lit.load_state_dict(sd)
    else:
        src_lit.model.load_state_dict(sd)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fisher = compute_fisher_weights(
        src_lit.model,
        img_dm.train_dataloader(),
        num_batches=int(fcfg.get("num_batches", 16)),
        device=device,
    )
    # Move back to CPU so the rest of dataset construction stays on CPU.
    fisher = {k: v.detach().cpu() for k, v in fisher.items()}
    print(
        f"[Fisher] computed over {fcfg.get('num_batches', 16)} batches "
        f"of {fcfg.image_dataset} (bs={fcfg.batch_size})"
    )
    return fisher


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    run_name = generate_slug()
    L.seed_everything(cfg.seed)

    accelerator, runtime_cfg = resolve_runtime(cfg.accelerator)

    index_bits = int(cfg.dataset.get("index_bits", 32))

    fisher_weights = _maybe_compute_fisher(cfg, runtime_cfg)

    datamodule = BlocksDataModule(
        source_checkpoint=cfg.dataset.source_checkpoint,
        rows=cfg.dataset.rows,
        cols=cfg.dataset.cols,
        batch_size=cfg.batch_size,
        index_bits=index_bits,
        runtime=runtime_cfg,
        fisher_weights=fisher_weights,
    )
    datamodule.setup()

    block_size = int(cfg.dataset.rows) * int(cfg.dataset.cols)

    checkpoint_dir, unique_run_name, init_ckpt_path = setup_checkpoint_dir(
        log_dir=cfg.log_dir,
        dataset_name=cfg.dataset.name,
        prefix=cfg.prefix,
        run_name=run_name,
    )

    model = LitIndexToBlock(
        block_size=block_size,
        index_bits=index_bits,
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
        epochs=cfg.epochs,
        rel_eps=float(cfg.model.get("rel_eps", 1e-6)),
        arch=cfg.model.get("arch", "mlp"),
        hidden_dim=int(cfg.model.get("hidden_dim", 256)),
        depth=int(cfg.model.get("depth", 5)),
        w0_first=float(cfg.model.get("w0_first", 30.0)),
        w0_hidden=float(cfg.model.get("w0_hidden", 1.0)),
        save_init_path=init_ckpt_path,
    )

    has_logger = cfg.get("wandb", {}).get("enabled", False)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best",
        monitor="train/loss",
        mode="min",
        save_top_k=1,
        save_last=False,
    )
    callbacks: list[L.Callback] = [checkpoint_callback]
    if has_logger:
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    if has_logger:
        wandb_cfg = cfg.wandb
        wandb_run_name = wandb_cfg.get("name") or unique_run_name
        logger = WandbLogger(
            project=wandb_cfg.get("project", "momos-reproduction"),
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

    trainer.fit(model, datamodule=datamodule)

    if checkpoint_callback.best_model_path:
        print(f"\nBest checkpoint: {checkpoint_callback.best_model_path}")
        if checkpoint_callback.best_model_score is not None:
            print(f"Best train/loss: {checkpoint_callback.best_model_score:.6f}")


if __name__ == "__main__":
    main()
