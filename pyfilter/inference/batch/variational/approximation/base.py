import torch
from torch.distributions import Distribution
from torch.nn import Module
from abc import ABC
from ....utils import Process


class BaseApproximation(Module, ABC):
    """
    Abstract base class for constructing variational approximations.
    """

    def __init__(self):
        super().__init__()

    def initialize(self, data: torch.Tensor, model: Process):
        """
        Method to be overridden by derived classes. Given the data to use for fitting, together with the model,
        initialize the required parameters/attributes of ``self``.

        Args:
            data: The data to consider, of size ``(number of observations, [dimension of observation space])``.
            model: The model to consider.
        """

        raise NotImplementedError()

    def get_approximation(self) -> Distribution:
        """
        Method to be overridden by derived classes. Returns the distribution of the variational approximation.
        """

        raise NotImplementedError()
