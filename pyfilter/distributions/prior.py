from torch.distributions import TransformedDistribution, biject_to
import torch
from typing import Tuple
from torch.nn import Module
from ..mixins import DistributionBuilderMixin
from .typing import HyperParameters, DistributionOrBuilder


class Prior(DistributionBuilderMixin, Module):
    """
    Class representing a Bayesian prior on a parameter.
    """

    def __init__(self, base_dist: DistributionOrBuilder, **parameters: HyperParameters):
        super().__init__()

        self.base_dist = base_dist
        parameters["validate_args"] = parameters.pop("validate_args", False)

        for k, v in parameters.items():
            self.register_buffer(k, v if isinstance(v, torch.Tensor) else torch.tensor(v))

        self.shape = self().event_shape

    @property
    def bijection(self):
        return biject_to(self().support)

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

        return self.unconstrained_prior.log_prob(self.get_unconstrained(x))

    def get_numel(self, constrained=True):
        return (self().event_shape if not constrained else self.unconstrained_prior.event_shape).numel()

    def get_slice_for_parameter(self, prev_index, constrained=True) -> Tuple[slice, int]:
        numel = self.get_numel(constrained)

        return slice(prev_index, prev_index + numel), numel
