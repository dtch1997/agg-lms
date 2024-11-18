import abc
import torch
import torch.nn as nn

from torch import Tensor
from jaxtyping import Float, Int

Tokens = Int[Tensor, "..."]
ResidualActs = Float[Tensor, "... n_layer d_model"]
LogitActs = Float[Tensor, "... d_vocab"]
Loss = Float[Tensor, "()"]


class Backbone(abc.ABC, nn.Module):
    """Abstract base class for a backbone"""

    d_model: int
    n_layer: int

    def __init__(self, d_model: int, n_layer: int):
        super().__init__()
        self.d_model = d_model
        self.n_layer = n_layer

    @abc.abstractmethod
    def get_residual_acts(self, tokens: Tokens) -> ResidualActs:
        """Get the residual activations per layer for a given input"""
        pass


class Predictor(abc.ABC, nn.Module):
    """Abstract base class for a predictor"""

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
        """Compute logits from residual activations

        NOTE: Needs to be differentiable!"""
        pass


class Model(nn.Module):
    backbone: Backbone
    predictor: Predictor

    def __init__(self, backbone: Backbone, predictor: Predictor):
        super().__init__()
        self.backbone = backbone
        self.predictor = predictor

    def forward(self, tokens: Tokens) -> LogitActs:
        residual_acts = self.backbone.get_residual_acts(tokens)
        return self.predictor.get_prediction(residual_acts)


class Dataset(abc.ABC, torch.utils.data.Dataset):

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    # NOTE: Dataset returns tokens
    @abc.abstractmethod
    def __getitem__(self, idx) -> Tokens:
        pass
