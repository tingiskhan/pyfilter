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
        new_loc = old.mean.clone()
        new_scale = old.stddev.clone()

        new_loc[indices] = new.mean[indices]
        new_scale[indices] = new.stddev[indices]

        old.base_dist.__init__(new_loc, new_scale)

