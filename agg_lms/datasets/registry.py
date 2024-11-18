from datasets import load_dataset, concatenate_datasets
from agg_lms.datasets.huggingface import HFDataset


def make_c4_code_dataset() -> HFDataset:
    # NOTE: both datasets only have the "train" split
    c4_ds = load_dataset("NeelNanda/c4-tokenized-2b", split="train")
    code_ds = load_dataset("NeelNanda/code-tokenized", split="train")

    mixed_ds = concatenate_datasets([c4_ds, code_ds])  # type: ignore
    return HFDataset(mixed_ds)
