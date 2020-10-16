from .meanfield import StateMeanField
import torch
from torch.distributions import LowRankMultivariateNormal


class StateLowRank(StateMeanField):
    def __init__(self, model):
        if model.ndim > 0:
            raise NotImplementedError(f"Maximum ")

        super().__init__(model)
        self._w = None

    def initialize(self, data, *args):
        self._mean = torch.zeros((data.shape[0] + 1, *self._model.increment_dist.event_shape), requires_grad=True)
        self._log_std = torch.zeros_like(self._mean, requires_grad=True)
        self._w = torch.zeros((self._mean.shape[0], 2), requires_grad=True)

        return self

    def dist(self):
        return LowRankMultivariateNormal(self._mean, self._w, self._log_std.exp())

    def get_parameters(self):
        return self._mean, self._log_std, self._w