import torch
from torch.distributions import LowRankMultivariateNormal, Independent
from torch.nn import Parameter
from .meanfield import StateMeanField


class StateLowRank(StateMeanField):
    """
    State approximation using low rank matrices as per ``torch.distributions.LowRankMultivariateNormal``.
    """

    def __init__(self, rank: int = 2):
        """
        Initializes the ``StateLowRank`` class.

        Args:
            rank: The rank of the matrix to use, see ``torch.distributions.LowRankMultivariateNormal``.
        """

        super().__init__()
        self.w = None
        self._rank = rank
        self._dim = None

    def initialize(self, data, model):
        shape = model.hidden.increment_dist().event_shape

        mean = torch.zeros((data.shape[0] + 1, *shape))
        log_std = torch.zeros_like(mean)

        self.mean = Parameter(mean, requires_grad=True)
        self.log_std = Parameter(log_std, requires_grad=True)

        self.w = Parameter(torch.zeros((mean.shape[0], self._rank, self._rank)), requires_grad=True)
        self._dim = model.hidden.n_dim

        return self

    def get_approximation(self):
        return Independent(LowRankMultivariateNormal(self.mean, self.w, self.log_std.exp()), self._dim)
