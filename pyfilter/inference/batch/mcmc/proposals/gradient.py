from torch.distributions import Independent, Normal
import torch
from math import sqrt
from .random_walk import RandomWalk
from ....utils import params_to_tensor, eval_prior_log_prob, params_from_tensor
from .....timeseries import NewState


class GradientBasedProposal(RandomWalk):
    """
    Implements a proposal utilizing gradients.
    """

    def __init__(self, eps: float = 1e-4):
        super().__init__(sqrt(2 * eps))
        self._eps = eps

    def build(self, state, filter_, y):
        smoothed = filter_.smooth(state.filter_state.states)

        params = params_to_tensor(filter_.ssm, constrained=False)
        params.requires_grad_(True)

        params_from_tensor(filter_.ssm, params)

        time = torch.stack(tuple(s.x.time_index for s in state.filter_state.states))

        xtm1 = NewState(time[:-1], values=smoothed[:-1])
        xt = NewState(time[1:], values=smoothed[1:])

        y = y.view(y.shape[0], 1, 1, *y.shape[1:])

        hidden_dens = filter_.ssm.hidden.build_density(xtm1)
        obs_dens = filter_.ssm.observable.build_density(xt)

        logl = filter_.ssm.hidden.initial_dist.log_prob(smoothed[0]).mean(-1)
        logl += eval_prior_log_prob(filter_.ssm, constrained=False).squeeze(-1)
        logl = (hidden_dens.log_prob(xt.values) + obs_dens.log_prob(y)).mean(-1).sum(0)

        logl.backward(torch.ones_like(logl))

        loc = params + params.grad * self._eps
        params.detach_()

        # TODO: Better?
        for v in filter_.ssm.parameters():
            v.detach_()

        return Independent(Normal(loc, self._scale), 1)