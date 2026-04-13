from torchvision import datasets, transforms

import utils.init as utils


def build_transform(dataset_name, img_size, is_train):
    """Build image preprocessing/augmentation transforms.

    Args:
        dataset_name: Dataset key, one of ``cifar10``, ``mnist``, ``fashion_mnist``.
        img_size: Final square image size used by the model.
        is_train: When True, include train-time augmentation.

    Returns:
        A ``torchvision.transforms.Compose`` object.
    """
    if dataset_name in ["mnist", "fashion_mnist"]:
        ops = []
        if img_size != 28:
            ops.append(transforms.Resize((img_size, img_size)))
        ops.extend([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        return transforms.Compose(ops)

    if dataset_name == "cifar10":
        ops = []
        if is_train:
            ops.extend(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4),
                ]
            )
        if img_size != 32:
            ops.append(transforms.Resize((img_size, img_size)))
        if is_train:
            ops.append(transforms.RandAugment(num_ops=2, magnitude=9))
        ops.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
                ),
            ]
        )
        return transforms.Compose(ops)

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def load_dataset(dataset_name, is_train, transform, data_dir):
    """Load one supported torchvision dataset split.

    Args:
        dataset_name: Dataset key, one of ``cifar10``, ``mnist``, ``fashion_mnist``.
        is_train: If True loads the training split, else test split.
        transform: Transform pipeline to apply to samples.
        data_dir: Root directory where dataset files are stored/downloaded.

    Returns:
        A torchvision dataset instance.
    """
    if dataset_name == "cifar10":
        return datasets.CIFAR10(
            data_dir, train=is_train, transform=transform, download=True
        )
    if dataset_name == "mnist":
        return datasets.MNIST(
            data_dir, train=is_train, transform=transform, download=True
        )
    if dataset_name == "fashion_mnist":
        return datasets.FashionMNIST(
            data_dir, train=is_train, transform=transform, download=True
        )
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def count_from_pct(total, pct, pct_name):
    """Convert split ratio/percentage to an absolute sample count.

    Args:
        total: Total number of available samples.
        pct: Ratio in ``(0, 1]`` or percent in ``(0, 100]``. ``None`` means disabled.
        pct_name: Human-readable field label used in error messages.

    Returns:
        Integer count, or ``None`` when ``pct`` is ``None``.
    """
    pct = utils.normalize_pct(pct, pct_name)
    if pct is None:
        return None
    count = int(total * pct)
    if count <= 0:
        raise ValueError(f"{pct_name} is too small and yields zero samples")
    return count
