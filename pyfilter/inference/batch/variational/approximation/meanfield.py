import torch
from torch.distributions import Independent, Normal, TransformedDistribution, Distribution
from torch.nn import Parameter
from typing import Tuple
from .base import BaseApproximation
from ....utils import priors_from_model


class StateMeanField(BaseApproximation):
    """
    Mean field approximation for state.
    """

    def __init__(self):
        super().__init__()
        self._dim = None

        self.register_parameter("mean", None)
        self.register_parameter("log_std", None)

    def initialize(self, data, model):
        mean = torch.zeros((data.shape[0] + 1, *model.hidden.increment_dist().event_shape))
        log_std = torch.zeros_like(mean)

        self.mean = Parameter(mean, requires_grad=True)
        self.log_std = Parameter(log_std, requires_grad=True)

        self._dim = model.hidden.n_dim

        return self

    def dist(self) -> Distribution:
        return Independent(Normal(self.mean, self.log_std.exp()), self._dim + 1)

    def get_inferred_states(self) -> torch.Tensor:
        return self.mean


class ParameterMeanField(BaseApproximation):
    """
    Mean field approximation for parameters.
    """

    def __init__(self):
        super().__init__()
        self._bijections = None
        self._mask = None

        self.register_parameter("mean", None)
        self.register_parameter("log_std", None)

    def get_parameters(self):
        return self.mean, self.log_std

    def initialize(self, data, model):
        self._bijections = tuple()
        self._mask = tuple()
        means = tuple()

        left = 0
        for p in priors_from_model(model):
            slc, numel = p.get_slice_for_parameter(left, False)

            val = p.bijection.inv(p().mean)
            if val.dim() == 0:
                val.unsqueeze_(-1)

            means += (val,)
            self._bijections += (p.bijection,)
            self._mask += (slc,)

            left += numel

        mean = torch.cat(means)
        log_std = torch.zeros_like(mean)

        self.mean = Parameter(mean, requires_grad=True)
        self.log_std = Parameter(log_std, requires_grad=True)

        return self

    def dist(self) -> Distribution:
        return Independent(Normal(self.mean, self.log_std.exp()), 1)

    def get_transformed_dists(self) -> Tuple[Distribution, ...]:
        res = tuple()
        for bij, msk in zip(self._bijections, self._mask):
            dist = TransformedDistribution(Normal(self.mean[msk], self.log_std[msk].exp()), bij)
            res += (dist,)

        return res
