from abc import ABC
from torch.distributions import Distribution
import torch
from torch.nn import Parameter
from functools import lru_cache
from typing import Tuple, Union, Callable, Optional
from .base import Base
from ..distributions import DistributionWrapper, Prior
from ..typing import ShapeLike, ArrayType
from .state import TimeseriesState
from ..utils import size_getter


DistOrBuilder = DistributionWrapper


class StochasticProcess(Base, ABC):
    def __init__(
        self,
        parameters: Tuple[ArrayType, ...],
        initial_dist: Optional[DistributionWrapper],
        increment_dist: DistributionWrapper,
        initial_transform: Union[Callable[[Distribution, Tuple[torch.Tensor, ...]], Distribution], None] = None,
    ):
        """
        The base class for time series.

        :param parameters: The parameters governing the dynamics
        :param initial_dist: The initial distribution
        :param increment_dist: The distribution of the increments
        """

        super().__init__()

        self.initial_dist = initial_dist
        self.init_transform = initial_transform
        self.increment_dist = increment_dist

        self._input_dim = self.n_dim

        for i, p in enumerate(parameters):
            name = f"parameter_{i}"

            if isinstance(p, Prior):
                self.register_prior(name, p)
            elif isinstance(p, Parameter):
                self.register_parameter(name, p)
            else:
                self.register_buffer(name, p if isinstance(p, torch.Tensor) else torch.tensor(p))

    def functional_parameters(self) -> Tuple[Parameter, ...]:
        res = dict()
        res.update(self._parameters)
        res.update(self._buffers)

        return tuple(v for _, v in sorted(res.items(), key=lambda k: k[0]))

    @property
    @lru_cache()
    def n_dim(self):
        """
        Returns the dimension of the process. If it's univariate it returns a 0, 1 for a vector etc - just like torch.
        """

        return len((self.initial_dist or self.increment_dist)().event_shape)

    @property
    @lru_cache()
    def num_vars(self) -> int:
        """
        Returns the number of variables of the stochastic process. E.g. if it's a univariate process it returns 1, and
        if it's a multivariate process it return the number of elements in the vector or matrix.
        """

        dist = self.initial_dist or self.increment_dist
        return dist().event_shape.numel()

    def parameters_to_array(self, constrained=True, as_tuple=False):
        res = tuple(
            (p if constrained else prior.get_unconstrained(p)).view(-1, prior.get_numel(constrained))
            for p, prior in self.parameters_and_priors()
        )

        if not res or as_tuple:
            return res

        return torch.cat(res, dim=-1)

    def parameters_from_array(self, array, constrained=True):
        tot_shape = sum(p.get_numel(constrained) for p in self.priors())

        if array.shape[-1] != tot_shape:
            raise ValueError(f"Shapes not congruent, {array.shape[-1]} != {tot_shape}")

        left = 0
        for p, prior in self.parameters_and_priors():
            slc, numel = prior.get_slice_for_parameter(left, constrained)
            p.update_values(array[..., slc], prior, constrained)

            left += numel

        return self

    def define_initial_density(self, shape: ShapeLike = None) -> Distribution:
        """
        Defines and returns the initial density.
        """

        dist = self.initial_dist().expand(size_getter(shape))

        if self.init_transform is not None:
            dist = self.init_transform(dist, *self.functional_parameters())

        return dist

    def initial_sample(self, shape: ShapeLike = None) -> TimeseriesState:
        """
        Samples a state from the initial distribution.
        """

        return TimeseriesState(0.0, self.define_initial_density(shape).sample())

    def sample_path(self, steps, samples=None, x_s=None) -> torch.Tensor:
        x_s = self.initial_sample(samples) if x_s is None else x_s

        res = (x_s,)
        for i in range(1, steps):
            res += (self.propagate(res[-1]),)

        return torch.stack(tuple(r.state for r in res), dim=0)

    def propagate_conditional(self, x: TimeseriesState, u: torch.Tensor, parameters=None) -> TimeseriesState:
        """
        Propagate the process conditional on both state and draws from incremental distribution.

        :param x: The current or previous state, depending on whether it's a hidden or observable process
        :param u: The current draws from the incremental distribution
        :param parameters: Whether to override the parameters that go into the functions with some other values
        """

        return self.propagate_state(self._propagate_conditional(x, u, parameters=parameters), x)

    def _propagate_conditional(self, x: TimeseriesState, u: torch.Tensor, parameters=None) -> torch.Tensor:
        raise NotImplementedError()

    def prop_apf(self, x: TimeseriesState) -> TimeseriesState:
        """
        Method used by APF. Propagates the state one step forward.

        :param x: The previous state
        :return: The new state
        """

        raise NotImplementedError()
