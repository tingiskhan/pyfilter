from torch.distributions import Distribution
import torch
from copy import deepcopy
from typing import TypeVar, Callable
from torch.nn import Module
from ..distributions import PriorMixin, DistributionWrapper
from .state import TimeseriesState
from ..typing import ShapeLike
from ..utils import size_getter


T = TypeVar("T")

# TODO: Move functionality from process.py to here
# TODO: Rename this to process and remove process.py
class Base(PriorMixin, Module):
    """
    Defines the base class for stochastic processes
    """

    def __init__(self, initial_dist: DistributionWrapper):
        super().__init__()
        self.inital_dist = initial_dist
        self._post_process_state: Callable[[TimeseriesState, TimeseriesState], None] = None

    @property
    def n_dim(self) -> int:
        """
        Returns the dimension of the process. If it's univariate it returns a 0, 1 for a vector etc - just like torch.
        """

        return len(self.initial_dist().event_shape)

    @property
    def num_vars(self) -> int:
        """
        Returns the number of variables of the stochastic process. E.g. if it's a univariate process it returns 1, and
        if it's a multivariate process it return the number of elements in the vector or matrix.
        """

        return self.initial_dist().event_shape.numel()

    def build_initial_density(self, shape: ShapeLike = None) -> Distribution:
        """
        Defines and returns the initial density.
        """

        return self.initial_dist().expand(size_getter(shape))

    def initial_sample(self, shape: ShapeLike = None) -> TimeseriesState:
        """
        Samples a state from the initial distribution.
        """

        return TimeseriesState(0.0, self.build_initial_density(shape).sample())

    def log_prob(self, y: torch.Tensor, x: TimeseriesState) -> torch.Tensor:
        """
        Evaluate the log likelihood for y | x. Depending on whether the process is an observable or not, `x` and `y`
        take on different meanings.

        :param y: If observable corresponds to observed data, else the process value at x_t
        :param x: If observable corresponds to process value at x_t, else timeseries value at x_{t-1}
        """

        dist = self.build_density(x)

        return dist.log_prob(y)

    def build_density(self, x: TimeseriesState) -> Distribution:
        """
        Method for defining the density used in `propagate`. Differs whether it's an observable or hidden process. If
        it's an observable process this method corresponds to the observation density, whereas for a hidden process it
        corresponds to the transition density.

        :param x: The current or previous state of the hidden process, depending on whether self is observable or hidden
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

        dist = self.build_density(x)

        return self.propagate_state(dist.sample(), x)

    def sample_path(self, steps: int, samples: ShapeLike = None, x_s: TimeseriesState = None) -> torch.Tensor:
        """
        Samples a trajectory from the model.

        :param steps: The number of steps
        :param samples: Number of sample paths
        :param x_s: The start value for the latent process
        :return: An array of sampled values
        """

        x_s = self.initial_sample(samples) if x_s is None else x_s

        res = (x_s,)
        for i in range(1, steps):
            res += (self.propagate(res[-1]),)

        return torch.stack(tuple(r.state for r in res), dim=0)

    def copy(self):
        """
        Returns a deep copy of the object.
        """

        return deepcopy(self)
