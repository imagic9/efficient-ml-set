"""CIFAR-10 data pipeline: train / val / test splits and augmentations."""
from torch.utils.data import DataLoader, random_split
import torch
from torchvision import datasets, transforms

# CIFAR-10 channel statistics (computed over the training set).
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def _train_transform():
    # Standard CIFAR augmentation: random crop with reflection padding + horizontal flip.
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])


def _eval_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])


def build_loaders(data_dir: str, batch_size: int = 256, num_workers: int = 8,
                  val_size: int = 5000, seed: int = 42, shuffle_seed=None):
    """Return (train_loader, val_loader, test_loader).

    The 50k training images are split into train / val; the 10k official test
    set is kept untouched for the final report number.

    `seed` fixes the train/val split (kept identical across every run so the test
    set identity never changes). `shuffle_seed`, if given, drives the train
    DataLoader's shuffle via its own Generator, so the batch order depends *only*
    on that seed -- not on how much RNG the model construction happened to consume.
    This lets two different architectures see the same data order at a given seed
    (a controlled comparison), and makes multi-seed averaging reproducible.
    """
    train_full = datasets.CIFAR10(data_dir, train=True, download=True,
                                  transform=_train_transform())
    # Second view of the same files but with eval-time transforms for validation.
    val_full = datasets.CIFAR10(data_dir, train=True, download=True,
                                transform=_eval_transform())
    test_set = datasets.CIFAR10(data_dir, train=False, download=True,
                                transform=_eval_transform())

    n = len(train_full)
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen).tolist()
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]

    train_set = torch.utils.data.Subset(train_full, train_idx)
    val_set = torch.utils.data.Subset(val_full, val_idx)

    common = dict(num_workers=num_workers, pin_memory=True)
    gen = torch.Generator().manual_seed(shuffle_seed) if shuffle_seed is not None else None
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              drop_last=False, generator=gen, **common)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader
