""" Prediction strategies """
import torch
import torch.nn as nn

from einops import einsum
from torch import Tensor
from jaxtyping import Float
from agg_lms.core import Predictor, ResidualActs, LogitActs

class LastLayerPredictor(Predictor):
    """ Predict the logits from the last layer's residual activations """

    def __init__(self, d_model: int, n_layer: int, d_vocab: int):
        super().__init__(d_model, n_layer, d_vocab)
        self.unembedding = nn.Linear(self.d_model, self.d_vocab)
        self.layernorm = nn.LayerNorm(self.d_model)

    def get_prediction(self, residual_acts: ResidualActs) -> LogitActs:
        normed_residual_acts = self.layernorm(residual_acts[..., -1, :])
        return self.unembedding(normed_residual_acts)
    
class WeightedPredictor(Predictor):
    """ Predict the logits from the last layer's residual activations """

    def __init__(self, d_model: int, n_layer: int, d_vocab: int, coefficients: Float[Tensor, "n_layer"]):
        super().__init__(d_model, n_layer, d_vocab)
        self.unembedding = nn.Linear(self.d_model, self.d_vocab)
        self.layernorm = nn.LayerNorm(self.d_model)
        self.coefficients = coefficients

    @property 
    def normalised_coefficients(self):
        return self.coefficients / self.coefficients.sum()

    def get_prediction(self, residual_acts: ResidualActs) -> LogitActs:
        normed_residual_acts = self.layernorm(residual_acts)
        weighted_residual_acts = einsum(
            self.normalised_coefficients, 
            normed_residual_acts,
            "n_layer, ... n_layer d_model -> ... d_model", 
        )
        return self.unembedding(weighted_residual_acts)
    

class WeightedPerLayerPredictor(Predictor):
    def __init__(self, d_model: int, n_layer: int, d_vocab: int, coefficients: Float[Tensor, "n_layer"]):
        super().__init__(d_model, n_layer, d_vocab)
        self.unembedding = nn.Linear(self.d_model, self.d_vocab)
        self.adapters = nn.Parameter(torch.randn(n_layer, d_model, d_model))
        # TODO: orthogonal init? 
        self.coefficients = coefficients

    @property 
    def normalised_coefficients(self):
        return self.coefficients / self.coefficients.sum()
    
    def get_prediction(self, residual_acts: ResidualActs) -> LogitActs:
        normed_residual_acts = self.layernorm(residual_acts)
        # Weight the layer-wise activations
        weighted_residual_acts = einsum(
            self.normalised_coefficients, 
            normed_residual_acts,
            "n_layer, ... n_layer d_model -> ... d_model", 
        )
        # Apply the layer-wise adapters
        # This is functionally equivalent to having a different unembedding matrix for each layer
        adapted_residual_acts = einsum(
            self.adapters,
            weighted_residual_acts,
            "... n_layer d_model, ... n_layer d_model -> ... n_layer d_model",
        )
        return self.unembedding(adapted_residual_acts)