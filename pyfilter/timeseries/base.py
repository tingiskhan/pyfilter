from torch.distributions import Distribution
import torch
from copy import deepcopy
from typing import Tuple, Union, TypeVar
from ..utils import ShapeLike
from ..prior_module import PriorModule
from .timeseriesstate import TimeseriesState
from ..typing import StateLike


T = TypeVar("T")


class Base(PriorModule):
    def parameters_to_array(self, transformed=False, as_tuple=False) -> torch.Tensor:
        raise NotImplementedError()

    def parameters_from_array(self, array: torch.Tensor, transformed=False):
        raise NotImplementedError()

    def sample_params(self, shape: ShapeLike):
        """
        Samples the parameters of the model in place.
        """

        for param, prior in self.parameters_and_priors():
            param.sample_(prior, shape)

        return self

    def log_prob(self, y: torch.Tensor, x: TimeseriesState) -> torch.Tensor:
        """
        Weights the process of the current state `x_t` with the previous `x_{t-1}`.
        :param y: The value at x_t
        :param x: The value at x_{t-1}
        """

        dist = self.define_density(x)

        return dist.log_prob(y)

    def define_density(self, x: TimeseriesState) -> Distribution:
        raise NotImplementedError()

    def build_state(self, new_values: torch.Tensor, prev_state: StateLike):
        return TimeseriesState(prev_state.time_index + 1, new_values)

    def propagate(self, x: TimeseriesState, as_dist=False) -> StateLike:
        """
        Propagates the model forward conditional on the previous state and current parameters.
        :param x: The previous state
        :param as_dist: Whether to return the new value as a distribution
        :return: Samples from the model
        """

        dist = self.define_density(x)

        if as_dist:
            return dist

        return self.build_state(dist.sample(), x)

    def sample_path(
        self, steps: int, samples: Union[int, Tuple[int, ...]] = None, x_s: TimeseriesState = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Samples a trajectory from the model.
        :param steps: The number of steps
        :param samples: Number of sample paths
        :param x_s: The start value for the latent process
        :return: An array of sampled values
        """

        raise NotImplementedError()

    def eval_prior_log_prob(self, constrained=True) -> torch.Tensor:
        """
        Calculates the prior log-likelihood of the current values of the parameters.
        :param constrained: If you use an unconstrained proposal you need to use `transformed=True`
        """

        return sum((prior.eval_prior(p, constrained) for p, prior in self.parameters_and_priors()))

    def copy(self):
        """
        Returns a deep copy of the object.
        :return: Copy of current instance
        """
        res = deepcopy(self)

        return res
