from .meanfield import StateMeanField
import torch
from torch.distributions import LowRankMultivariateNormal


class StateLowRank(StateMeanField):
    def __init__(self, rank: int = 2):
        super().__init__()
        self._w = None
        self._rank = rank

    def initialize(self, data, model, *args):
        self._mean = torch.zeros((data.shape[0] + 1, *model.increment_dist.event_shape), requires_grad=True)
        self._log_std = torch.zeros_like(self._mean, requires_grad=True)
        self._w = torch.zeros((self._mean.shape[0], self._rank), requires_grad=True)

        return self

    def dist(self):
        return LowRankMultivariateNormal(self._mean, self._w, self._log_std.exp())

    def get_parameters(self):
        return self._mean, self._log_std, self._w
