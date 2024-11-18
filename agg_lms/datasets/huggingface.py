import torch
from datasets import Dataset as RawHFDataset
from agg_lms.core import Tokens, Dataset


class HFDataset(Dataset):
    """Dataset wrapper for HuggingFace datasets

    Assumes that the dataset has a "tokens" field with the tokenized input
    """

    def __init__(self, dataset: RawHFDataset, max_length=1024):
        self.dataset = dataset
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx) -> Tokens:
        tokens = torch.tensor(self.dataset[idx]["tokens"][: self.max_length])
        return tokens
