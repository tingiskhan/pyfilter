from .base import BaseApproximation
import torch
from torch.distributions import Independent, Normal
from ..utils import stacker


class StateMeanField(BaseApproximation):
    def __init__(self, model):
        """
        Implements a mean field approximation of the state space.
        :param model: The model
        :type model: pyfilter.timeseries.base.StochasticProcess
        """

        super().__init__()
        self._mean = None
        self._log_std = None
        self._model = model

    def initialize(self, data, *args):
        self._mean = torch.zeros((data.shape[0] + 1, *self._model.increment_dist.event_shape), requires_grad=True)
        self._log_std = torch.ones_like(self._mean, requires_grad=True)

        return self

    def dist(self):
        return Independent(Normal(self._mean, self._log_std.exp()), self._model.ndim + 1)

    def get_parameters(self):
        return self._mean, self._log_std


# TODO: Only supports 1D parameters currently
class ParameterMeanField(BaseApproximation):
    def __init__(self):
        """
        Implements the mean field for parameters.
        """

        super().__init__()
        self._mean = None
        self._log_std = None

    def get_parameters(self):
        return self._mean, self._log_std

    def initialize(self, parameters, *args):
        stacked = stacker(parameters, lambda u: u.t_values)

        self._mean = torch.zeros(stacked.concated.shape[1:], device=stacked.concated.device)
        self._log_std = torch.ones_like(self._mean)

        for p, msk in zip(parameters, stacked.mask):
            try:
                self._mean[msk] = p.bijection.inv(p.distr.mean)
            except NotImplementedError:
                pass

        self._mean.requires_grad_(True)
        self._log_std.requires_grad_(True)

        return self

    def dist(self):
        return Independent(Normal(self._mean, self._log_std.exp()), 1)