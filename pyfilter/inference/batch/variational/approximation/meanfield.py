import torch
from torch.distributions import Independent, Normal, TransformedDistribution, Distribution
from torch.nn import Parameter
from typing import Tuple
from .base import BaseApproximation


class MeanField(BaseApproximation):
    """
    Base class for mean field approximations, i.e. approximations in which the posterior distribution is modelled using
    independent normal distributions:
        .. math::
            p(x_1, \\dots, x_n) = \\prod_{i=1}^n \\mathcal{N}(x_i \\mid \\mu_i, \\sigma_i).
    """

    def __init__(self):
        """
        Initializes the ``MeanField`` class.
        """

        super().__init__()

        self.mean = None
        self.log_std = None
        self._independent_dim = None

    def initialize(self, shape):
        mean = torch.zeros(shape)
        log_std = torch.zeros_like(mean)

        self.mean = Parameter(mean, requires_grad=True)
        self.log_std = Parameter(log_std, requires_grad=True)

        self._independent_dim = len(shape)

        return self

    def get_approximation(self) -> Distribution:
        return Independent(Normal(self.mean, self.log_std.exp()), self._independent_dim)
