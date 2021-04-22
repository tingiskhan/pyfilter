import torch
from torch.distributions import Distribution
from torch.nn import Module
from abc import ABC
from ....utils import Process


class BaseApproximation(Module, ABC):
    def __init__(self):
        """
        Base class for constructing variational approximations.
        """
        super().__init__()

    def initialize(self, data: torch.Tensor, model: Process):
        return self

    def dist(self) -> Distribution:
        raise NotImplementedError()

    def entropy(self) -> torch.Tensor:
        """
        Returns the entropy of the variational approximation
        """

        return self.dist().entropy()
