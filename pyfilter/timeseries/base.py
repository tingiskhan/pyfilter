from torch.distributions import Distribution
import torch
from copy import deepcopy
from typing import Tuple, Union, TypeVar, Callable
from torch.nn import Module
from ..prior_mixin import PriorMixin
from .state import TimeseriesState
from ..typing import ShapeLike


T = TypeVar("T")


class Base(PriorMixin, Module):
    def __init__(self):
        super().__init__()
        self._post_process_state: Callable[[TimeseriesState, TimeseriesState], None] = None

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
        Depending on whether the process is an observable or not, `x` and `y` take on different meanings.

        :param y: If observable corresponds to observed data, else the process value at x_t
        :param x: If observable corresponds to process value at x_t, else timeseries value at x_{t-1}
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

    def propagate_state(self, new_values: torch.Tensor, prev_state: TimeseriesState, time_increment=1.0):
        new_state = TimeseriesState(prev_state.time_index + time_increment, new_values)

        if self._post_process_state is not None:
            self._post_process_state(new_state, prev_state)

        return new_state

    # TODO: Perhaps use forward instead and register forward hook?
    def register_state_post_process(self, func: Callable[[TimeseriesState, TimeseriesState], None]):
        self._post_process_state = func

    def propagate(self, x: TimeseriesState) -> TimeseriesState:
        """
        Propagates the model forward conditional on the previous state and current parameters.

        :param x: The previous state
        :return: Samples from the model
        """

        dist = self.define_density(x)

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
        """

        return deepcopy(self)
