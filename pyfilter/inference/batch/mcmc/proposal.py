from torch.distributions import Independent, Normal
import torch
from .state import PMMHState
from ....filters import BaseFilter
from ...utils import params_to_tensor, eval_prior_log_prob, parameters_and_priors_from_model, params_from_tensor
from ....timeseries import NewState


class IndependentProposal(object):
    def __init__(self, scale: float = 1e-2):
        self._scale = scale

    def __call__(self, state: PMMHState, filter_: BaseFilter, y: torch.Tensor):
        return Independent(Normal(params_to_tensor(filter_.ssm, constrained=False), self._scale), 1)


class GradientBasedProposal(IndependentProposal):
    """
    Implements a proposal utilizing gradients.
    """

    def __init__(self, eps: float = 1e-4, **kwargs):
        super().__init__(**kwargs)
        self._eps = eps

    def __call__(self, state: PMMHState, filter_: BaseFilter, y: torch.Tensor):
        smoothed = filter_.smooth(state.filter_result.states)

        params = params_to_tensor(filter_.ssm, constrained=False)
        params.requires_grad_(True)

        params_from_tensor(filter_.ssm, params)

        time = torch.stack(tuple(s.x.time_index for s in state.filter_result.states))

        xtm1 = NewState(time[:-1], values=smoothed[:-1])
        xt = NewState(time[1:], values=smoothed[1:])

        y = y.view(y.shape[0], 1, 1, *y.shape[1:])

        hidden_dens = filter_.ssm.hidden.build_density(xtm1)
        obs_dens = filter_.ssm.observable.build_density(xt)

        logl = (hidden_dens.log_prob(xt.values) + obs_dens.log_prob(y)).mean(-1).sum(0) + eval_prior_log_prob(filter_.ssm, constrained=False).squeeze(-1)
        logl.backward(torch.ones_like(logl))

        loc = params + params.grad * self._eps
        params.detach_()

        # TODO: Better?
        for v in filter_.ssm.parameters():
            v.detach_()

        return Independent(Normal(loc, self._scale), 1)