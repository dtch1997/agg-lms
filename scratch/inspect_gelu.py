from transformer_lens import HookedTransformer


model = HookedTransformer.from_pretrained("gelu-2l")
print(model.cfg.d_vocab)

model = HookedTransformer.from_pretrained("gelu-4l")
print(model.cfg.d_vocab)

from datasets import load_dataset

ds = load_dataset("NeelNanda/c4-tokenized-2b")
print(ds)

from datasets import load_dataset

ds = load_dataset("NeelNanda/c4-tokenized-2b", split="train")
print(ds.take(1)["tokens"])

ds = load_dataset("NeelNanda/code-tokenized", split="train")
print(ds.take(1)["tokens"])

# Print average sequence length
from datasets import load_dataset

ds = load_dataset("NeelNanda/c4-tokenized-2b", split="train")
print(len(ds))
c4_len = len(ds)
ds = ds.take(1_000)
print(sum(len(x["tokens"]) for x in ds) / len(ds))

# Print average sequence length
from datasets import load_dataset

ds = load_dataset("NeelNanda/code-tokenized", split="train")
print(len(ds))
code_len = len(ds)
ds = ds.take(1_000)
print(sum(len(x["tokens"]) for x in ds) / len(ds))

print(c4_len / (c4_len + code_len))
