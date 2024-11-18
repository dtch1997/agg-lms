import abc 
import torch.nn as nn

from torch import Tensor
from jaxtyping import Float, Int

Tokens = Int[Tensor, "..."]
ResidualActs = Float[Tensor, "... n_layer d_model"]
LogitActs = Float[Tensor, "... d_vocab"]

class Backbone(abc.ABC, nn.Module):
    """ Abstract base class for a backbone """

    d_model: int
    n_layer: int

    def __init__(self, d_model: int, n_layer: int):
        super().__init__()
        self.d_model = d_model
        self.n_layer = n_layer
    
    @abc.abstractmethod
    def get_residual_acts(self, tokens: Tokens) -> ResidualActs:
        """ Get the residual activations per layer for a given input """
        pass

class Predictor(abc.ABC, nn.Module):
    """ Abstract base class for a predictor """

    d_model: int
    n_layer: int
    d_vocab: int

    def __init__(self, d_model: int, n_layer: int, d_vocab: int):
        super().__init__()
        self.d_model = d_model
        self.n_layer = n_layer
        self.d_vocab = d_vocab

    @abc.abstractmethod
    def get_prediction(self, residual_acts: ResidualActs) -> LogitActs:
        """ Compute logits from residual activations 
        
        NOTE: Needs to be differentiable! """
        pass

class Model(abc.ABC, nn.Module):
    backbone: Backbone
    predictor: Predictor