from .base import BaseApproximation
import torch
from torch.distributions import Independent, Normal
from ..kernels import stacker


class StateMeanField(BaseApproximation):
    def __init__(self):
        """
        Implements a mean field approximation of the state space.
        """
        super().__init__()
        self._mean = None
        self._std = None

    def initialize(self, data, ndim):
        self._mean = torch.zeros((data.shape[0] + 1, ndim), requires_grad=True)
        self._std = torch.ones_like(self._mean, requires_grad=True)

        # ===== Start optimization ===== #
        self._dist = Independent(Normal(self._mean, self._std), 2)

        return self

    def get_parameters(self):
        return self._mean, self._std


# TODO: Only supports 1D parameters currently
class ParameterMeanField(BaseApproximation):
    def __init__(self):
        """
        Implements the mean field for parameters.
        """

        super().__init__()
        self._mean = None
        self._std = None

    def get_parameters(self):
        return self._mean, self._std

    def initialize(self, parameters, *args):
        self._mean = torch.zeros(sum(p.c_numel() for p in parameters))
        self._std = torch.ones_like(self._mean)

        _, mask = stacker(parameters)

        for p, msk in zip(parameters, mask):
            self._mean[msk] = p.bijection.inv(p.distr.mean)

        self._mean.requires_grad_(True)
        self._std.requires_grad_(True)

        self._dist = Independent(Normal(self._mean, self._std), 1)

        return self