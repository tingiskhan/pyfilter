from torch.distributions import Independent, Normal
import torch
from .state import PMMHState
from ....filters import BaseFilter
from ...utils import params_to_tensor


class IndependentProposal(object):
    def __init__(self, scale=1e-2):
        self._scale = scale

    def __call__(self, state: PMMHState, filter_: BaseFilter, y: torch.Tensor):
        return Independent(Normal(params_to_tensor(filter_.ssm, constrained=False), self._scale), 1)


class GradientBasedProposal(IndependentProposal):
    def __call__(self, state: PMMHState, filter_: BaseFilter, y: torch.Tensor):
        x = torch.stack(tuple(s.x for s in state.filter_result.states), dim=0)
        w = torch.stack(tuple(s.w for s in state.filter_result.states), dim=0)

        for p in filter_.ssm.hidden.trainable_parameters:
            p.requires_grad_(True)

        xtm1 = x[:-1]
        xt = x[1:]
        y = y.view(y.shape[0], 1, 1, *y.shape[1:])

        # TODO: Calculate smoothing recursion, should be able to vectorize
        # TODO: Add prior to estimate
