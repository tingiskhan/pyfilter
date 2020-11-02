import numpy as np
import torch
from functools import lru_cache
from collections import OrderedDict
from copy import deepcopy
from torch.distributions import Distribution, TransformedDistribution, Transform, biject_to
from typing import Union, Tuple


TensorOrDist = Union[torch.Tensor, Distribution]
ArrayType = Union[float, int, TensorOrDist, np.ndarray]


def size_getter(shape: Union[int, Tuple[int, ...], torch.Size]) -> torch.Size:
    """
    Helper function for defining a size object.
    :param shape: The shape
    :return: Size object
    """

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
    # NB: This line exists only for backwards compatibility; the
    # general expectation is that backward_hooks is an empty
    # OrderedDict.  See Note [Don't serialize hooks]
    param._backward_hooks = backward_hooks
    param._prior = prior

    return param


class Parameter(torch.Tensor):
    def __new__(cls, parameter: ArrayType = None, requires_grad=False):
        if isinstance(parameter, Parameter):
            raise ValueError(f"The input cannot be of instance '{parameter.__class__.__name__}'!")
        elif isinstance(parameter, torch.Tensor):
            _data = parameter
        elif isinstance(parameter, Distribution):
            # This is just a place holder
            _data = torch.empty(parameter.event_shape)
        else:
            _data = torch.tensor(parameter) if not isinstance(parameter, np.ndarray) else torch.from_numpy(parameter)

        return torch.Tensor._make_subclass(cls, _data, requires_grad)

    def __init__(self, parameter: ArrayType = None, requires_grad=False):
        """
        The parameter class.
        """
        self._prior = parameter if isinstance(parameter, Distribution) else None

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]

        result = type(self)(self.data.clone(), self.requires_grad)
        result._prior = deepcopy(self.prior)

        memo[id(self)] = result
        return result

    @property
    @lru_cache()
    def transformed_dist(self):
        """
        Returns the unconstrained distribution.
        """

        if not self.trainable:
            raise ValueError('Is not of `Distribution` instance!')

        return TransformedDistribution(self.prior, self.bijection.inv)

    @property
    def bijection(self) -> Transform:
        """
        Returns a bijected function for transforms from unconstrained to constrained space.
        """
        if not self.trainable:
            raise ValueError("Is not of `Distribution` instance!")

        return biject_to(self.prior.support)

    @property
    def values(self) -> torch.Tensor:
        """
        Returns the actual values of the parameters.
        """

        return self.data

    @values.setter
    def values(self, x: torch.Tensor):
        """
        Sets the values of x.
        :param x: The values
        """
        if not isinstance(x, torch.Tensor):
            raise ValueError("Is not the same type!")
        elif x.shape != self.data.shape:
            raise ValueError("Not of same shape!")
        elif not self.trainable:
            self.data[:] = x
            return

        support = self._prior.support.check(x)

        if (~support).any():
            raise ValueError("Found values outside bounds!")

        self.data[:] = x

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
        """
        Sets transformed values.
        :param x: The values
        """

        self.values = self.bijection(x)

    @property
    def prior(self):
        """
        Returns the distribution.
        """

        if self.trainable:
            return self._prior

        raise ValueError("Does not have a distribution!")

    def sample_(self, shape: Union[int, Tuple[int, ...], torch.Size] = None):
        """
        Samples the variable from prior distribution in place.
        :param shape: The shape to use
        """
        if not self.trainable:
            raise ValueError("Cannot initialize parameter as it is not of instance 'Distribution'!")

        new_data = self._prior.sample(size_getter(shape))

        if self.data.shape == new_data.shape:
            self.values = new_data
        else:
            self.data = new_data

        return self

    @property
    def trainable(self):
        """
        Boolean of whether parameter is trainable.
        """

        return isinstance(self._prior, Distribution)

    def __reduce_ex__(self, protocol):
        return (
            _rebuild_parameter,
            (self.data, self.requires_grad, self._prior, OrderedDict())
        )
