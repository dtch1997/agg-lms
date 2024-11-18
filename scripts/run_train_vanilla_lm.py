from agg_lms.core import Model
from agg_lms.backbone.gpt import GPT, GPTConfig
from agg_lms.predictor import LastLayerPredictor
from agg_lms.train import LanguageModelTrainingModule, train_model

if __name__ == "__main__":

    gpt_config = GPTConfig()

    model = Model(
        backbone=GPT(config=gpt_config),
        predictor=LastLayerPredictor(
            d_model=gpt_config.n_embd,
            n_layer=gpt_config.n_layer,
            d_vocab=gpt_config.vocab_size,
        )
    )

    lightning_module = LanguageModelTrainingModule(
        model=model,
        learning_rate=1e-4,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        warmup_iters=2000,
        lr_decay_iters=600000,
        min_lr=1e-6,
    )

    # TODO: implement train, val datasets
    train_dataset = None
    val_dataset = None

    train_model(
        lightning_module,
        train_dataset, 
        val_dataset, 
        max_epochs=10,
        batch_size=64,
        num_workers=4,
    )