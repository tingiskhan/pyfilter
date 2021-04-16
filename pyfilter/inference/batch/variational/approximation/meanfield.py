from .base import BaseApproximation
import torch
from torch.distributions import Independent, Normal, TransformedDistribution, Distribution
from .....timeseries import StochasticProcess
from .....distributions import Prior
from typing import Tuple


class StateMeanField(BaseApproximation):
    """
    Mean field approximation for state.
    """

    def __init__(self):
        super().__init__()
        self.mean = None
        self.log_std = None
        self._dim = None

    def initialize(self, data, model: StochasticProcess, *args):
        self.mean = torch.zeros((data.shape[0] + 1, *model.increment_dist().event_shape), requires_grad=True)
        self.log_std = torch.zeros_like(self.mean, requires_grad=True)
        self._dim = model.n_dim

        return self

    def dist(self):
        return Independent(Normal(self.mean, self.log_std.exp()), self._dim + 1)

    def get_parameters(self):
        return self.mean, self.log_std

    def get_inferred_states(self) -> torch.Tensor:
        return self.mean


# TODO: Only supports 1D parameters currently
class ParameterMeanField(BaseApproximation):
    """
    Mean field approximation for parameters.
    """

    def __init__(self):
        super().__init__()
        self.mean = None
        self.log_std = None
        self._bijections = None
        self._mask = None

    def get_parameters(self):
        return self.mean, self.log_std

    def initialize(self, priors: Tuple[Prior, ...], *args):
        self._bijections = tuple()

        means = tuple()
        self._mask = tuple()

        left = 0
        for p in priors:
            slc, numel = p.get_slice_for_parameter(left, False)

            means += (p.bijection.inv(p().mean),)
            self._bijections += (p.bijection,)
            self._mask += (slc,)

            left += numel

        self.mean = torch.stack(means)
        self.mean.requires_grad_(True)

        self.log_std = torch.zeros_like(self.mean, requires_grad=True)

        return self

    def dist(self):
        return Independent(Normal(self.mean, self.log_std.exp()), 1)

    def get_transformed_dists(self) -> Tuple[Distribution, ...]:
        res = tuple()
        for bij, msk in zip(self._bijections, self._mask):
            dist = TransformedDistribution(Normal(self.mean[msk], self.log_std[msk].exp()), bij)
            res += (dist,)

        return res
