from torch.distributions import TransformedDistribution, biject_to
import torch
from typing import Tuple
from torch.nn import Module
from .mixin import BuilderMixin


class Prior(BuilderMixin, Module):
    def __init__(self, base_dist, **parameters):
        super().__init__()

        self.base_dist = base_dist

        for k, v in parameters.items():
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)

            self.register_buffer(k, v)

        self.bijection = biject_to(self().support)
        self.shape = self().event_shape

    @property
    def unconstrained_prior(self):
        return TransformedDistribution(self(), self.bijection.inv)

    def get_unconstrained(self, x: torch.Tensor):
        return self.bijection.inv(x)

    def get_constrained(self, x: torch.Tensor):
        return self.bijection(x)

    def eval_prior(self, x: torch.Tensor, constrained=True) -> torch.Tensor:
        if constrained:
            return self().log_prob(x)

        return self.unconstrained_prior.log_prob(self.get_constrained(x))

    def get_numel(self, constrained=True):
        return (self().event_shape if not constrained else self.unconstrained_prior.event_shape).numel()

    def get_slice_for_parameter(self, prev_index, constrained=True) -> Tuple[slice, int]:
        numel = self.get_numel(constrained)

        return slice(prev_index, prev_index + numel), numel
