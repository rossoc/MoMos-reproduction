"""Train a single-hidden-layer MLP: block-index -> motif-id (classification).

Consumes the ``motifs.pt`` artifact produced by ``src/extract_motifs.py`` and
fits ``LitIndexToMotif``. Inputs are binary-encoded indices, targets are motif
ids in [0, K).
"""

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, TensorDataset
from coolname import generate_slug

from model.lit_index_to_motif import LitIndexToMotif, encode_indices_bits
from utils.init import setup_checkpoint_dir


@hydra.main(config_path="configs", config_name="config_motifs", version_base="1.3")
def main(cfg: DictConfig):
    run_name = generate_slug()
    L.seed_everything(cfg.seed)

    blob = torch.load(cfg.motifs_path, map_location="cpu")
    assignments = blob["assignments"].long()  # (N,)
    layer_ids = blob["layer_ids"].long()  # (N,) in [0, L)
    within_layer_idx = blob["within_layer_idx"].long()  # (N,)
    num_layers = int(blob["num_layers"])
    n_blocks = int(blob["num_blocks"])
    n_motifs = int(blob["num_motifs"])
    max_within = int(within_layer_idx.max().item()) + 1
    print(
        f"[motifs] N={n_blocks} blocks, K={n_motifs} motifs, "
        f"L={num_layers} layers, max within-layer idx={max_within} "
        f"(rows={blob['rows']}, cols={blob['cols']})"
    )

    index_bits = int(cfg.model.index_bits)
    if (1 << index_bits) < max_within:
        raise ValueError(
            f"index_bits={index_bits} too small for within-layer max={max_within} "
            f"(need >= {max_within.bit_length()})"
        )

    bits = encode_indices_bits(within_layer_idx, n_bits=index_bits)  # (N, B)
    layer_onehot = torch.nn.functional.one_hot(
        layer_ids, num_classes=num_layers
    ).float()  # (N, L)
    x = torch.cat([layer_onehot, bits], dim=1)  # (N, L+B)
    input_dim = x.size(1)
    dataset = TensorDataset(x, assignments)

    train_ds = dataset

    train_dl = DataLoader(
        train_ds,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    val_dl = None

    checkpoint_dir, unique_run_name, init_ckpt_path = setup_checkpoint_dir(
        log_dir=cfg.log_dir,
        dataset_name="motifs",
        prefix=cfg.prefix,
        run_name=run_name,
        model_name=cfg.model.get("arch", "mlp"),
    )

    model = LitIndexToMotif(
        num_motifs=n_motifs,
        input_dim=input_dim,
        hidden_dim=int(cfg.model.hidden_dim),
        learning_rate=float(cfg.model.learning_rate),
        weight_decay=float(cfg.model.weight_decay),
        epochs=int(cfg.epochs),
        arch=str(cfg.model.get("arch", "mlp")),
        depth=int(cfg.model.get("depth", 5)),
        w0_first=float(cfg.model.get("w0_first", 30.0)),
        w0_hidden=float(cfg.model.get("w0_hidden", 1.0)),
        save_init_path=init_ckpt_path,
    )

    monitor = "val/acc" if val_dl is not None else "train/acc"
    ckpt_cb = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best",
        monitor=monitor,
        mode="max",
        save_top_k=1,
        save_last=False,
    )
    callbacks: list[L.Callback] = [ckpt_cb]

    has_logger = cfg.get("wandb", {}).get("enabled", False)
    if has_logger:
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))
        wcfg = cfg.wandb
        logger = WandbLogger(
            project=wcfg.get("project", "momos-reproduction"),
            entity=wcfg.get("entity", None),
            name=wcfg.get("name") or unique_run_name,
            tags=list(wcfg.get("tags", [])),
            log_model=wcfg.get("log_model", False),
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
    )
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    if ckpt_cb.best_model_path:
        print(f"\nBest checkpoint: {ckpt_cb.best_model_path}")
        if ckpt_cb.best_model_score is not None:
            print(f"Best {monitor}: {float(ckpt_cb.best_model_score):.6f}")


if __name__ == "__main__":
    main()
