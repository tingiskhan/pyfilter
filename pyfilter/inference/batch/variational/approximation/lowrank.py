import torch
from torch.distributions import LowRankMultivariateNormal
from torch.nn import Parameter
from .meanfield import StateMeanField


class StateLowRank(StateMeanField):
    """
    State approximation using low rank matrices.
    """

    def __init__(self, rank: int = 2):
        super().__init__()
        self._w = None
        self._rank = rank

    def initialize(self, data, model):
        mean = torch.zeros((data.shape[0] + 1, *model.hidden.increment_dist().event_shape), requires_grad=True)
        log_std = torch.zeros_like(mean, requires_grad=True)

        self.mean = Parameter(mean)
        self.log_std = Parameter(log_std)

        self._w = torch.zeros((mean.shape[0], self._rank, self._rank), requires_grad=True)

        return self

    def dist(self):
        return LowRankMultivariateNormal(self.mean, self._w, self.log_std.exp())

    def entropy(self):
        return super(StateLowRank, self).entropy().sum(0)
