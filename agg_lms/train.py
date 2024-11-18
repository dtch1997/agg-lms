import pytorch_lightning as pl
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple
from agg_lms.core import Model, Tokens, LogitActs, Loss

def build_lr_scheduler(
    optimizer, 
    warmup_iters: int, 
    lr_decay_iters: int, 
    learning_rate: float, 
    min_lr: float
):

    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return min_lr + coeff * (learning_rate - min_lr)
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

class LanguageModelTrainingModule(pl.LightningModule):
    def __init__(
        self,
        model: Model,
        learning_rate: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.01,
        warmup_iters: float = 2000,
        lr_decay_iters: float = 600000,
        min_lr: float = 1e-6,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.betas = betas
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = lr_decay_iters
        self.min_lr = min_lr

    def _compute_loss(self, tokens: Tokens) -> Loss:
        logits = self.model(tokens)
        logits = logits[:, :-1]
        target_tokens = tokens[:, 1:]
        loss = F.cross_entropy(
            logits,
            target_tokens,
            ignore_index=-100,
            reduction="mean"
        )
        return loss
    
    def training_step(self, batch: Tokens, batch_idx: int) -> Loss:
        loss = self._compute_loss(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch: Tokens, batch_idx: int) -> Loss:
        loss = self._compute_loss(batch)
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def configure_optimisers(self):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.model.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': self.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        # TODO: Report number of decayed and nondecayed params in logging
        # num_decay_params = sum(p.numel() for p in decay_params)
        # num_nodecay_params = sum(p.numel() for p in nodecay_params)
        # print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        # print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=self.betas, fused = True)

        return {
            "optimizer": optimizer,
            "lr_scheduler": build_lr_scheduler(
                optimizer, 
                self.warmup_iters, 
                self.lr_decay_iters, 
                self.learning_rate, 
                self.min_lr,
            )
        }

def train_model(
    lightning_module: LanguageModelTrainingModule,
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset | None = None,
    max_epochs: int = 10,
    batch_size: int = 32,
    num_workers: int = 4,
):
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        gradient_clip_val=1.0
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
    
    trainer.fit(lightning_module, train_loader, val_loader)
    return lightning_module