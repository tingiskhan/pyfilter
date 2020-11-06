import numpy as np
import torch
from functools import lru_cache
from collections import OrderedDict
from copy import deepcopy
from torch.distributions import Distribution, TransformedDistribution, Transform, biject_to
from typing import Union, Tuple
from ..utils import ShapeLike

TensorOrDist = Union[torch.Tensor, Distribution]
ArrayType = Union[float, int, TensorOrDist, np.ndarray]


def size_getter(shape: ShapeLike) -> torch.Size:
    if shape is None:
        return torch.Size([])
    elif isinstance(shape, torch.Size):
        return shape
    elif isinstance(shape, int):
        return torch.Size([shape])

    return torch.Size(shape)


# NB: This is basically the same as original, but we include the prior as well
def _rebuild_parameter(data, requires_grad, prior, backward_hooks):
    param = Parameter(data, requires_grad)
    param._backward_hooks = backward_hooks
    param._prior = prior

    return param


class Parameter(torch.Tensor):
    _prior = None

    def __new__(cls, parameter: ArrayType = None, requires_grad=False):
        if isinstance(parameter, Parameter):
            return parameter
        elif isinstance(parameter, torch.Tensor):
            _data = parameter
        elif isinstance(parameter, Distribution):
            _data = torch.empty(parameter.event_shape)
        else:
            _data = torch.tensor(parameter) if not isinstance(parameter, np.ndarray) else torch.from_numpy(parameter)

        return torch.Tensor._make_subclass(cls, _data, requires_grad)

    def __init__(self, parameter: ArrayType = None, requires_grad=False):
        if isinstance(parameter, Distribution):
            self._prior = parameter
        elif isinstance(parameter, Parameter):
            self._prior = parameter.prior

        self.requires_grad = requires_grad

    # Same as torch
    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]

        result = Parameter(self.data.clone(), self.requires_grad)

        if self._prior is not None:
            result._prior = deepcopy(self.prior)

        memo[id(self)] = result
        return result

    @property
    def batch_shape(self):
        dim = len(self.prior.event_shape)
        if dim == 0:
            return self.shape

        return self.shape[:-dim]

    @property
    def prior(self):
        if self.trainable:
            return self._prior

        raise ValueError("Does not have a distribution!")

    @property
    @lru_cache()
    def bijected_prior(self):
        if not self.trainable:
            raise ValueError('Is not of `Distribution` instance!')

        return TransformedDistribution(self.prior, self.bijection.inv)

    @property
    def bijection(self) -> Transform:
        """
        Returns the bijection from unconstrained to constrained space.
        """

        if not self.trainable:
            raise ValueError("Is not of 'Distribution' instance!")

        return biject_to(self.prior.support)

    @property
    def values(self) -> torch.Tensor:
        return self

    @values.setter
    def values(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise ValueError("Is not the same type!")
        elif x.shape != self.data.shape:
            raise ValueError("Not of same shape!")
        elif not self.trainable:
            self[:] = x
            return

        support = self._prior.support.check(x)

        if (~support).any():
            raise ValueError("Found values outside bounds!")

        self[:] = x

    @property
    def t_values(self) -> torch.Tensor:
        """
        Returns the transformed values.
        """

        if not self.trainable:
            raise ValueError("Cannot transform parameter not of instance 'Distribution'!")

        return self.bijection.inv(self.data)

    @t_values.setter
    def t_values(self, x: torch.Tensor):
        self.values = self.bijection(x)

    def sample_(self, shape: Union[int, Tuple[int, ...], torch.Size] = None):
        """
        Samples the variable from prior distribution in place.
        """

        if not self.trainable:
            raise ValueError("Cannot initialize parameter as it is not of instance 'Distribution'!")

        new_data = self._prior.sample(size_getter(shape))

        if self.shape == new_data.shape:
            self.values = new_data
        else:
            self.data = new_data

        return self

    @property
    def trainable(self):
        return isinstance(self._prior, Distribution)

    def numel_(self, transformed=True) -> int:
        return (self.prior.event_shape if not transformed else self.bijected_prior.event_shape).numel()

    def get_slice_for_parameter(self, prev_index, transformed=False) -> Tuple[Union[slice, int], int]:
        numel = self.numel_(transformed)

        if numel == 1:
            return prev_index, numel

        return slice(prev_index, prev_index + numel), numel

    def __reduce_ex__(self, protocol):
        return (
            _rebuild_parameter,
            (self.data, self.requires_grad, self._prior, OrderedDict())
        )
