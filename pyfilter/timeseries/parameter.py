import numpy as np
import torch
from functools import lru_cache
from scipy.stats import gaussian_kde
from collections import OrderedDict
from copy import deepcopy
from torch.distributions import Distribution, TransformedDistribution, Transform, biject_to
from typing import Union, Tuple


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
    def __new__(cls, parameter: Union[torch.Tensor, Distribution] = None, requires_grad=False):
        if isinstance(parameter, Parameter):
            raise ValueError('The input cannot be of instance `{}`!'.format(parameter.__class__.__name__))
        elif isinstance(parameter, torch.Tensor):
            _data = parameter
        elif not isinstance(parameter, Distribution):
            _data = torch.tensor(parameter) if not isinstance(parameter, np.ndarray) else torch.from_numpy(parameter)
        else:
            # This is just a place holder
            _data = torch.empty(parameter.event_shape)

        return torch.Tensor._make_subclass(cls, _data, requires_grad)

    def __init__(self, parameter: Union[torch.Tensor, Distribution] = None, requires_grad=False):
        """
        The parameter class.
        """
        self._prior = parameter if isinstance(parameter, Distribution) else None

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]

        result = type(self)(self.data.clone(), self.requires_grad)
        result._prior = deepcopy(self._prior)

        memo[id(self)] = result
        return result

    @property
    def c_shape(self):
        """
        Returns the dimension of the prior.
        """

        return self.distr.event_shape

    def c_numel(self):
        """
        Custom 'numel' function for the number of elements in the prior's event shape.
        """

        res = 1
        for it in self.c_shape:
            res *= it

        return res

    @property
    @lru_cache()
    def transformed_dist(self):
        """
        Returns the unconstrained distribution.
        """

        if not self.trainable:
            raise ValueError('Is not of `Distribution` instance!')

        return TransformedDistribution(self._prior, [self.bijection.inv])

    @property
    def bijection(self) -> Transform:
        """
        Returns a bijected function for transforms from unconstrained to constrained space.
        """
        if not self.trainable:
            raise ValueError('Is not of `Distribution` instance!')

        return biject_to(self._prior.support)

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
            raise ValueError('Is not the same type!')
        elif x.shape != self.data.shape:
            raise ValueError('Not of same shape!')
        elif not self.trainable:
            self.data[:] = x
            return

        support = self._prior.support.check(x)

        if (~support).any():
            raise ValueError('Found values outside bounds!')

        self.data[:] = x

    @property
    def t_values(self) -> torch.Tensor:
        """
        Returns the transformed values.
        """

        if not self.trainable:
            raise ValueError('Cannot transform parameter not of instance `Distribution`!')

        return self.bijection.inv(self.data)

    @t_values.setter
    def t_values(self, x: torch.Tensor):
        """
        Sets transformed values.
        :param x: The values
        """

        self.values = self.bijection(x)

    @property
    def distr(self):
        """
        Returns the distribution.
        """

        if self.trainable:
            return self._prior

        raise ValueError('Does not have a distribution!')

    def sample_(self, shape: Union[int, Tuple[int, ...], torch.Size] = None):
        """
        Samples the variable from prior distribution in place.
        :param shape: The shape to use
        """
        if not self.trainable:
            raise ValueError('Cannot initialize parameter as it is not of instance `Distribution`!')

        self.data = self._prior.sample(size_getter(shape))

        return self

    @property
    def trainable(self):
        """
        Boolean of whether parameter is trainable.
        """

        return isinstance(self._prior, Distribution)

    def get_kde(self, weights: torch.Tensor = None):
        """
        Constructs KDE of the discrete representation on the transformed space.
        :param weights: The weights to use
        :return: KDE object
        """
        array = self.t_values

        # ===== Perform transformation ===== #
        w = self.bijection.inv.log_abs_det_jacobian(self.values, array).exp()
        weights = (weights * w if weights is not None else w).cpu().numpy()

        return gaussian_kde(array.cpu().numpy(), weights=weights)

    def get_plottable(self, num=100, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gets the range and likelihood of the parameter as a `numpy.ndarray`. Used for plotting.
        :return: Range, likelihood
        """

        kde = self.get_kde(**kwargs)

        # ===== Gets the range to plot ===== #
        vals = kde.dataset
        low = vals.min()
        high = vals.max()

        while kde.logpdf(low) > np.log(1e-3):
            low -= 1e-2

        while kde.logpdf(high) > np.log(1e-3):
            high += 1e-2

        xeval = np.linspace(low, high, num)
        xrange_ = self.bijection(torch.from_numpy(xeval)).cpu().numpy()

        return xrange_, np.exp(kde.logpdf(xeval))

    def __reduce_ex__(self, protocol):
        return (
            _rebuild_parameter,
            (self.data, self.requires_grad, self._prior, OrderedDict())
        )
