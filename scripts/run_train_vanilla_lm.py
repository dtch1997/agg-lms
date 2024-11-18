import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset

from agg_lms.core import Model
from agg_lms.backbone.gpt import GPT, GPTConfig
from agg_lms.predictor import LastLayerPredictor
from agg_lms.train import LanguageModelTrainingModule
from agg_lms.datasets.registry import make_c4_code_dataset
from agg_lms.utils import train_val_test_split

from pytorch_lightning.loggers import Logger, WandbLogger


def train_model(
    lightning_module: LanguageModelTrainingModule,
    train_dataset: Dataset,
    val_dataset: Dataset | None = None,
    logger: Logger | None = None,
    max_epochs: int = 10,
    batch_size: int = 32,
    num_workers: int = 4,
):
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        gradient_clip_val=1.0,
        logger=logger,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

    trainer.fit(lightning_module, train_loader, val_loader)
    return lightning_module


if __name__ == "__main__":

    wandb_logger = WandbLogger(
        project="agg-lms",
        entity="dtch1997",
        group="vanilla-lm",
    )

    gpt_config = GPTConfig(
        # Hardcoded for c4_code dataset
        block_size=1024,
        vocab_size=48262,
        n_layer=2,
        n_head=8,
        n_embd=512,
    )

    dataset = make_c4_code_dataset()
    train_dataset, val_dataset, _ = train_val_test_split(dataset)

    model = Model(
        backbone=GPT(config=gpt_config),
        predictor=LastLayerPredictor(
            d_model=gpt_config.n_embd,
            n_layer=gpt_config.n_layer,
            d_vocab=gpt_config.vocab_size,
        ),
    )

    # NOTE: Warmup iters math
    # - We want to warmup for 300M tokens to match Gelu-2L
    # - 32 entries per batch * 1024 tokens per entry = 32k tokens per batch
    # - 300M tokens / 32k tokens per batch = 9375 batches
    # - So we need to warmup for 9375 batches
    lightning_module = LanguageModelTrainingModule(
        model=model,
        learning_rate=1e-4,
        betas=(0.9, 0.99),
        weight_decay=0.05,
        warmup_iters=9375,
    )

    # NOTE: Max epoch math
    # - We want to train for 22B max tokens to match Gelu-2L
    # - 400k entries x 1k tokens per entry = 0.4B tokens per epoch
    # - 22B tokens / 0.4B tokens per epoch = 55 epochs
    # - So we need to train for 55 epochs
    max_epochs = 55

    train_model(
        lightning_module,
        train_dataset,
        val_dataset=val_dataset,
        logger=wandb_logger,
        max_epochs=max_epochs,
        batch_size=32,
        num_workers=4,
    )
