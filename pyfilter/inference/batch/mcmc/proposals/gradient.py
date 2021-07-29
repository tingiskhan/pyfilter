from torch.distributions import Independent, Normal
import torch
from math import sqrt
from torch.autograd import grad
from .random_walk import RandomWalk
from ....utils import params_to_tensor, eval_prior_log_prob, params_from_tensor
from .....timeseries import NewState


class GradientBasedProposal(RandomWalk):
    """
    Implements a proposal utilizing gradients.
    """

    def __init__(self, eps: float = 1e-4, use_second_order: bool = False):
        super().__init__(sqrt(2 * eps))
        self._eps = eps
        self._use_second_order = use_second_order

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

        g = grad(logl, params, torch.ones_like(logl), create_graph=self._use_second_order)[-1]

        step = self._eps * torch.ones_like(params)
        scale = self._scale * torch.ones_like(params)

        if self._use_second_order:
            neg_inv_hess = -1.0 / grad(g, params, torch.ones_like(g))[-1]
            mask = neg_inv_hess > 0.0

            step[mask] *= neg_inv_hess[mask]
            scale[mask] = step[mask].pow(0.5)

        loc = params + step * g
        params.detach_()

        # TODO: Better?
        for v in filter_.ssm.parameters():
            v.detach_()

        return Independent(Normal(loc, scale), 1)