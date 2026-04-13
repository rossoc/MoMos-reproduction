"""LightningDataModule for CIFAR-10 and Fashion-MNIST datasets."""

from torch.utils.data import DataLoader, Subset
import lightning as L

from data.util import build_transform, load_dataset, count_from_pct
from utils.init import normalize_pct

import torch


class ImageDataModule(L.LightningDataModule):
    """Lightning DataModule.

    Args:
        dataset_name: Dataset key to load.
        batch_size: Batch size for all splits.
        img_size: Target image size before model input.
        val_pct: Validation fraction/percent sampled from training set.
        test_pct: Fraction/percent sampled from official test split.
        split_seed: RNG seed for reproducible subset sampling.
        runtime: Optional runtime config dict (workers, pinning, prefetch).
        data_dir: Dataset storage/download directory.
    """

    def __init__(
        self,
        dataset_name,
        batch_size,
        img_size,
        val_pct=None,
        test_pct=None,
        split_seed=10,
        runtime=None,
        data_dir="./data",
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.img_size = img_size
        self.val_pct = normalize_pct(val_pct, "val_pct")
        self.test_pct = normalize_pct(test_pct, "test_pct")
        self.split_seed = split_seed
        self.runtime = runtime or {
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "prefetch_factor": None,
        }
        self.data_dir = data_dir

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.split_info = None

    def setup(self, stage=None):
        if self.train_dataset is not None:
            return

        train_data = load_dataset(
            self.dataset_name,
            True,
            build_transform(self.dataset_name, self.img_size, True),
            self.data_dir,
        )
        train_eval_data = load_dataset(
            self.dataset_name,
            True,
            build_transform(self.dataset_name, self.img_size, False),
            self.data_dir,
        )
        test_data = load_dataset(
            self.dataset_name,
            False,
            build_transform(self.dataset_name, self.img_size, False),
            self.data_dir,
        )

        total_train = len(train_data)
        total_test = len(test_data)

        rng = torch.Generator().manual_seed(int(self.split_seed))
        perm_train = torch.randperm(total_train, generator=rng)
        perm_test = torch.randperm(total_test, generator=rng)

        val_count = count_from_pct(total_train, self.val_pct, "val_pct")

        if val_count is None:
            # No validation split from training data; use test set for validation.
            self.train_dataset = train_data
            eval_count = count_from_pct(
                total_test,
                self.test_pct if self.test_pct is not None else 1.0,
                "test_pct",
            )
            eval_idx = perm_test[:eval_count].tolist()
            self.val_dataset = Subset(test_data, eval_idx)
            self.test_dataset = Subset(test_data, eval_idx)
            split_mode = "train_plus_test_as_validation"
            has_proper_test = False
            resolved_train = total_train
            resolved_val = eval_count
            resolved_test = eval_count
        else:
            # Split training data into train/val.
            train_count = total_train - val_count
            if train_count <= 0:
                raise ValueError("val_pct leaves no training samples")
            val_idx = perm_train[:val_count].tolist()
            train_idx = perm_train[val_count:].tolist()

            # Train uses augmented transforms; val uses eval transforms.
            self.train_dataset = Subset(train_data, train_idx)
            self.val_dataset = Subset(train_eval_data, val_idx)

            test_count = count_from_pct(
                total_test,
                self.test_pct if self.test_pct is not None else 1.0,
                "test_pct",
            )
            test_idx = perm_test[:test_count].tolist()
            self.test_dataset = Subset(test_data, test_idx)

            split_mode = "train_val_from_train_test"
            has_proper_test = True
            resolved_train = train_count
            resolved_val = val_count
            resolved_test = test_count

        self.split_info = {
            "split_mode": split_mode,
            "has_proper_test": has_proper_test,
            "train_size": resolved_train,
            "val_size": resolved_val,
            "test_size": resolved_test,
            "val_pct": self.val_pct,
            "test_pct": self.test_pct,
            "split_seed": int(self.split_seed),
        }

    def _build_dataloader(self, dataset, shuffle=False) -> DataLoader:
        kwargs = {
            "batch_size": self.batch_size,
            "shuffle": shuffle,
            "num_workers": int(self.runtime["num_workers"]),
            "pin_memory": bool(self.runtime["pin_memory"]),
        }
        if kwargs["num_workers"] > 0:
            kwargs["persistent_workers"] = bool(self.runtime["persistent_workers"])
            if self.runtime["prefetch_factor"] is not None:
                kwargs["prefetch_factor"] = int(self.runtime["prefetch_factor"])
        return DataLoader(dataset, **kwargs)

    def train_dataloader(self):
        """Return training DataLoader."""
        return self._build_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        """Return validation DataLoader."""
        return self._build_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        """Return test DataLoader."""
        return self._build_dataloader(self.test_dataset, shuffle=False)
