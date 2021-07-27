from torch.distributions import Independent, Normal
from .base import BaseProposal
from ....utils import params_to_tensor


class RandomWalk(BaseProposal):
    """
    Implements a random walk proposal.
    """

    def __init__(self, scale: float = 1e-2):
        self._scale = scale

    def build(self, state, filter_, y):
        return Independent(Normal(params_to_tensor(filter_.ssm, constrained=False), self._scale), 1)

    def exchange(self, old, new, indices):
        new_loc = old.mean[:]
        new_loc[indices] = new.mean[indices]

        return Independent(Normal(new_loc, self._scale), 1)