from torch.distributions import Distribution
import torch
from copy import deepcopy
from typing import Tuple, Union, TypeVar, Callable
from ..prior_module import PriorModule
from .state import TimeseriesState
from ..typing import ShapeLike


T = TypeVar("T")


class Base(PriorModule):
    def __init__(self):
        super().__init__()
        self._post_process_state: Callable[[TimeseriesState, TimeseriesState], TimeseriesState] = None
        self._time_inc = 1

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
        """
        Method for defining the density used in `propagate`. Differs whether it's an observable or hidden process. If
        it's an observable process this method corresponds to the observation density, whereas for a hidden process it
        corresponds to the transition density.
        :param x: The current or previous state of the hidden state - depending on whether self is observable or hidden.
        """

        raise NotImplementedError()

    def propagate_state(self, new_values: torch.Tensor, prev_state: TimeseriesState):
        new_state = TimeseriesState(prev_state.time_index + self._time_inc, new_values)

        if self._post_process_state is None:
            return new_state

        return self._post_process_state(new_state, prev_state)

    def register_state_post_process(self, func: Callable[[TimeseriesState, TimeseriesState], TimeseriesState]):
        self._post_process_state = func

    def propagate(self, x: TimeseriesState, as_dist=False) -> Union[Distribution, TimeseriesState]:
        """
        Propagates the model forward conditional on the previous state and current parameters.
        :param x: The previous state
        :param as_dist: Whether to return the new value as a distribution
        :return: Samples from the model
        """

        dist = self.define_density(x)

        if as_dist:
            return dist

        return self.propagate_state(dist.sample(), x)

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
