from abc import ABC
from torch.distributions import Distribution
import torch
from torch.nn import Parameter
from functools import lru_cache
from .parameter import size_getter, ArrayType, TensorOrDist
from typing import Tuple, Union, Callable, Dict, Optional
from ..utils import ShapeLike
from .base import Base
from ..distributions import DistributionWrapper, Prior
from ..parameter import ExtendedParameter


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

        self._input_dim = self.ndim

        for i, p in enumerate(parameters):
            name = f"parameter_{i}"

            if isinstance(p, Prior):
                self.add_module(f"prior__{name}", p)
                self.register_parameter(name, ExtendedParameter(p, requires_grad=False))
            elif isinstance(p, Parameter):
                self.register_parameter(name, p)
            else:
                self.register_buffer(name, p if isinstance(p, torch.Tensor) else torch.tensor(p))

    def functional_parameters(self):
        res = dict()
        res.update(self._parameters)
        res.update(self._buffers)

        return tuple(v for _, v in sorted(res.items(), key=lambda k: k[0]))

    @property
    @lru_cache()
    def ndim(self):
        """
        Returns the dimension of the process. If it's univariate it returns a 0, 1 for a vector etc - just like torch.
        """

        return len((self.initial_dist or self.increment_dist)().event_shape)

    @property
    @lru_cache()
    def num_vars(self):
        """
        Returns the number of variables of the stochastic process. E.g. if it's a univariate process, it returns 1, and
        the number of elements in the vector/matrix.
        """

        dist = self.initial_dist or self.increment_dist
        return dist().event_shape.numel()

    def parameters_to_array(self, transformed=False, as_tuple=False):
        res = tuple(
            (p.values if not transformed else p.t_values).view(-1, p.numel_(transformed))
            for p in self.trainable_parameters
        )

        if not res or as_tuple:
            return res

        return torch.cat(res, dim=-1)

    def parameters_from_array(self, array, transformed=False):
        tot_shape = sum(p.numel_(transformed) for p in self.trainable_parameters)

        if array.shape[-1] != tot_shape:
            raise ValueError(f"Shapes not congruent, {array.shape[-1]} != {tot_shape}")

        left = 0
        for p in self.trainable_parameters:
            slc, numel = p.get_slice_for_parameter(left, transformed)

            if transformed:
                p.t_values = array[..., slc].reshape(p.t_values.shape)
            else:
                p.values = array[..., slc].reshape(p.shape)

            left += numel

        return self

    def i_sample(self, shape: ShapeLike = None, as_dist=False) -> TensorOrDist:
        """
        Samples from the initial distribution.
        :param shape: The number of samples
        :param as_dist: Whether to return the new value as a distribution
        :return: Samples from the initial distribution
        """

        dist = self.initial_dist().expand(size_getter(shape))

        if self.init_transform is not None:
            dist = self.init_transform(dist, *self.parameter_views)

        if as_dist:
            return dist

        return dist.sample()

    def sample_path(self, steps, samples=None, x_s=None, u=None) -> torch.Tensor:
        x_s = self.i_sample(samples) if x_s is None else x_s
        out = torch.zeros(steps, *x_s.shape, device=x_s.device, dtype=x_s.dtype)
        out[0] = x_s

        for i in range(1, steps):
            out[i] = self.propagate(out[i - 1])

        return out

    def propagate_u(self, x: torch.Tensor, u: torch.Tensor):
        """
        Propagate the process conditional on both state and draws from incremental distribution.
        :param x: The previous state
        :param u: The current draws from the incremental distribution
        """

        return self._propagate_u(x, u)

    def _propagate_u(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def prop_apf(self, x: torch.Tensor) -> torch.Tensor:
        """
        Method used by APF. Propagates the state one step forward.
        :param x: The previous state
        :return: The new state
        """

        raise NotImplementedError()

    def populate_state_dict(self):
        return {
            "_parameters": self._parameters,
            "_dist_builder": None if self._dist_builder is None else self._dist_builder.state_dict(),
        }
