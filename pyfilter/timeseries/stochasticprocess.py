from torch.distributions import Distribution
import torch
from copy import deepcopy
from typing import TypeVar, Callable, Union, Tuple
from torch.nn import Module, Parameter
from abc import ABC
from functools import lru_cache
from ..distributions import DistributionWrapper, Prior, PriorMixin
from .state import NewState
from ..typing import ShapeLike
from ..utils import size_getter


T = TypeVar("T")


class StochasticProcess(Module, ABC):
    """
    Defines the base class for stochastic processes
    """

    def __init__(
        self,
        initial_dist: DistributionWrapper,
        initial_transform: Union[Callable[["StochasticProcess", Distribution], Distribution], None] = None,
        num_steps: int = 1,
    ):
        super().__init__()
        self._initial_dist = initial_dist
        self._init_transform = initial_transform
        self.num_steps = num_steps

    @property
    @lru_cache(maxsize=None)
    def n_dim(self) -> int:
        """
        Returns the dimension of the process. If it's univariate it returns a 0, 1 for a vector etc - just like torch.
        """

        return len(self.initial_dist.event_shape)

    @property
    @lru_cache(maxsize=None)
    def num_vars(self) -> int:
        """
        Returns the number of variables of the stochastic process. E.g. if it's a univariate process it returns 1, and
        if it's a multivariate process it return the number of elements in the vector or matrix.
        """

        return self.initial_dist.event_shape.numel()

    @property
    def initial_dist(self) -> Distribution:
        dist = self._initial_dist()
        if self._init_transform is not None:
            dist = self._init_transform(self, dist)

        return dist

    def initial_sample(self, shape: ShapeLike = None) -> NewState:
        """
        Samples a state from the initial distribution.
        """

        dist = self.initial_dist

        return NewState(0.0, dist.expand(size_getter(shape)))

    def build_density(self, x: NewState) -> Distribution:
        """
        Method for defining the density used in `propagate`. Differs whether it's an observable or hidden process. If
        it's an observable process this method corresponds to the observation density, whereas for a hidden process it
        corresponds to the transition density.

        :param x: The current or previous state of the hidden process, depending on whether self is observable or hidden
        """

        raise NotImplementedError()

    def forward(self, x: NewState, time_increment=1.0) -> NewState:
        for _ in range(self.num_steps):
            density = self.build_density(x)
            x = x.propagate_from(dist=density, time_increment=time_increment)

        return x

    propagate = forward

    def sample_path(self, steps: int, samples: ShapeLike = None, x_s: NewState = None) -> torch.Tensor:
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

        return torch.stack(tuple(r.values for r in res), dim=0)

    def copy(self):
        """
        Returns a deep copy of the object.
        """

        return deepcopy(self)

    def propagate_conditional(self, x: NewState, u: torch.Tensor, parameters=None, time_increment=1.0) -> NewState:
        """
        Propagate the process conditional on both state and draws from incremental distribution.

        :param x: The current or previous state, depending on whether it's a hidden or observable process
        :param u: Draws from distribution
        :param parameters: Whether to override the parameters that go into the functions with some other values
        """

        raise NotImplementedError()


class StructuralStochasticProcess(PriorMixin, StochasticProcess, ABC):
    """
    Implements a stochastic process that has functional parameters, i.e. dynamics where the parameters directly
    influence the distribution.
    """

    def __init__(self, parameters, **kwargs):
        super().__init__(**kwargs)

        for i, p in enumerate(parameters):
            self._register_parameter_or_prior(f"parameter_{i}", p)

    def _register_parameter_or_prior(self, name: str, p):
        """
        Helper method for registering parameters or priors.
        """

        if isinstance(p, Prior):
            self.register_prior(name, p)
        elif isinstance(p, Parameter):
            self.register_parameter(name, p)
        else:
            self.register_buffer(name, p if isinstance(p, torch.Tensor) else torch.tensor(p))

    def functional_parameters(self, f: Callable[[torch.Tensor], torch.Tensor] = None) -> Tuple[Parameter, ...]:
        res = dict()
        res.update(self._parameters)
        res.update(self._buffers)

        return tuple(f(v) if f is not None else v for _, v in sorted(res.items(), key=lambda k: k[0]))
