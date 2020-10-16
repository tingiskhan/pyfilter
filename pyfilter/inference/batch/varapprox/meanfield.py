from .base import BaseApproximation
import torch
from torch.distributions import Independent, Normal, TransformedDistribution, Distribution
from ....utils import stacker
from ....timeseries import Parameter, StochasticProcess
from typing import Tuple


class StateMeanField(BaseApproximation):
    def __init__(self, model: StochasticProcess):
        super().__init__()
        self._mean = None
        self._log_std = None
        self._model = model

    def initialize(self, data, *args):
        self._mean = torch.zeros((data.shape[0] + 1, *self._model.increment_dist.event_shape), requires_grad=True)
        self._log_std = torch.zeros_like(self._mean, requires_grad=True)

        return self

    def dist(self):
        return Independent(Normal(self._mean, self._log_std.exp()), self._model.ndim + 1)

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
        self._stacked = None

    def get_parameters(self):
        return self._mean, self._log_std

    def initialize(self, parameters: Tuple[Parameter, ...], *args):
        self._stacked = stacked = stacker(parameters, lambda u: u.t_values)

        self._mean = torch.zeros(stacked.concated.shape[1:], device=stacked.concated.device)
        self._log_std = torch.ones_like(self._mean)

        self._bijections = tuple()
        for p, msk in zip(parameters, stacked.mask):
            if not p.trainable:
                continue

            self._mean[msk] = p.bijection.inv(p.distr.mean)
            self._bijections += (p.bijection,)

        self._mean.requires_grad_(True)
        self._log_std.requires_grad_(True)

        return self

    def dist(self):
        return Independent(Normal(self._mean, self._log_std.exp()), 1)

    def get_transformed_dists(self) -> Tuple[Distribution, ...]:
        res = tuple()
        for bij, msk in zip(self._bijections, self._stacked.mask):
            dist = TransformedDistribution(Normal(self._mean[msk], self._log_std[msk].exp()), bij)
            res += (dist,)

        return res