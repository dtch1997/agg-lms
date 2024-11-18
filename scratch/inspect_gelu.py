from transformer_lens import HookedTransformer


model = HookedTransformer.from_pretrained("gelu-2l")
print(model.cfg.d_vocab)

model = HookedTransformer.from_pretrained("gelu-4l")
print(model.cfg.d_vocab)