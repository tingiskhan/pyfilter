from .base import BaseApproximation
import torch
from torch.distributions import Independent, Normal, TransformedDistribution, Distribution
from .....timeseries import StochasticProcess
from .....distributions import Prior
from typing import Tuple


class StateMeanField(BaseApproximation):
    def __init__(self):
        super().__init__()
        self._mean = None
        self._log_std = None
        self._dim = None

    def initialize(self, data, model: StochasticProcess, *args):
        self._mean = torch.zeros((data.shape[0] + 1, *model.increment_dist.event_shape), requires_grad=True)
        self._log_std = torch.zeros_like(self._mean, requires_grad=True)
        self._dim = model.ndim

        return self

    def dist(self):
        return Independent(Normal(self._mean, self._log_std.exp()), self._dim + 1)

    def get_parameters(self):
        return self._mean, self._log_std

    def get_inferred_states(self) -> torch.Tensor:
        return self._mean


# TODO: Only supports 1D parameters currently
class ParameterMeanField(BaseApproximation):
    def __init__(self):
        super().__init__()
        self._mean = None
        self._log_std = None
        self._bijections = None
        self._mask = None

    def get_parameters(self):
        return self._mean, self._log_std

    def initialize(self, priors: Tuple[Prior, ...], *args):
        self._bijections = tuple()

        means = tuple()
        self._mask = tuple()

        left = 0
        for p in priors:
            slc, numel = p.get_slice_for_parameter(left, True)

            means += (p.bijection.inv(p.prior.mean),)
            self._bijections += (p.bijection,)
            self._mask += (slc,)

            left += numel

        self._mean = torch.stack(means)
        self._mean.requires_grad_(True)

        self._log_std = torch.zeros_like(self._mean, requires_grad=True)

        return self

    def dist(self):
        return Independent(Normal(self._mean, self._log_std.exp()), 1)

    def get_transformed_dists(self) -> Tuple[Distribution, ...]:
        res = tuple()
        for bij, msk in zip(self._bijections, self._mask):
            dist = TransformedDistribution(Normal(self._mean[msk], self._log_std[msk].exp()), bij)
            res += (dist,)

        return res
