""" Script to prepare the C4 + code dataset """

from datasets import load_dataset, concatenate_datasets
from sklearn.model_selection import train_test_split


def build_mixed_dataset():
    # NOTE: both datasets only have the "train" split
    c4_ds = load_dataset("NeelNanda/c4-tokenized-2b", split="train")
    code_ds = load_dataset("NeelNanda/code-tokenized", split="train")

    mixed_ds = concatenate_datasets([c4_ds, code_ds])  # type: ignore
    return mixed_ds


def split_dataset(
    ds,
    seed: int = 0,
):
    # Make a 40-10-50 train-val-test split
    trainval_ds, test_ds = train_test_split(
        ds,
        train_size=0.5,
        random_state=seed,
        shuffle=True,
    )
    train_ds, val_ds = train_test_split(
        trainval_ds,
        train_size=0.8,
        random_state=seed,
        shuffle=True,
    )

    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    ds = build_mixed_dataset()
    train_ds, val_ds, test_ds = split_dataset(ds)
