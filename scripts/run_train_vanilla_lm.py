from agg_lms.core import Model
from agg_lms.backbone.gpt import GPT, GPTConfig
from agg_lms.predictor import LastLayerPredictor


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