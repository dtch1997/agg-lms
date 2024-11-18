

from datasets import load_dataset

ds = load_dataset("NeelNanda/c4-tokenized-2b", split="train")
print(ds.take(1)['tokens'])