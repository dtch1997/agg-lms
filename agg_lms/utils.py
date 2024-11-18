import torch

from torch.utils.data import random_split
from agg_lms.core import Dataset


def print_gpu_utilization():
    """Print GPU utilization"""
    print(torch.cuda.memory_summary(device="cuda", abbreviated=False))


def train_val_test_split(
    ds: Dataset,
    train_frac: float = 0.4,
    test_frac: float = 0.5,
    seed: int = 0,
) -> tuple[Dataset, Dataset, Dataset]:
    """Make a train-val-test split

    Default is a 40-10-50 split.
    """
    train_len = int(train_frac * len(ds))
    test_len = int(test_frac * len(ds))
    val_len = len(ds) - train_len - test_len

    train_ds, val_ds, test_ds = random_split(
        ds,
        [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(seed),
    )

    return train_ds, val_ds, test_ds  # type: ignore
