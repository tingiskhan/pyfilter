import torch
from torch.distributions import Distribution
from torch.nn import Module
from abc import ABC


class BaseApproximation(Module, ABC):
    """
    Abstract base class for constructing variational approximations.
    """

    def __init__(self):
        super().__init__()

    def initialize(self, shape: torch.Size):
        """
        Method to be overridden by derived classes. Initializes the required attributes of the approximation given the
        shape and the model.

        Args:
            shape: The shape of the resulting approximation.
        """

        raise NotImplementedError()

    def get_approximation(self) -> Distribution:
        """
        Method to be overridden by derived classes. Returns the distribution of the variational approximation.
        """

        raise NotImplementedError()
