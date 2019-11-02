from torch import Tensor, distributions as dist
import numpy as np
import torch
from functools import lru_cache
from scipy.stats import gaussian_kde
from collections import OrderedDict
from copy import deepcopy


def size_getter(shape):
    """
    Helper function for defining a size object.
    :param shape: The shape
    :type shape: int|tuple[int]
    :return: Size object
    :rtype: torch.Size
    """

    return torch.Size([]) if shape is None else torch.Size(shape if isinstance(shape, (tuple, list)) else (shape,))


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
    def __new__(cls, parameter=None, requires_grad=False):
        if isinstance(parameter, Parameter):
            raise ValueError('The input cannot be of instance `{}`!'.format(parameter.__class__.__name__))
        elif isinstance(parameter, torch.Tensor):
            _data = parameter
        elif not isinstance(parameter, dist.Distribution):
            _data = torch.tensor(parameter) if not isinstance(parameter, np.ndarray) else torch.from_numpy(parameter)
        else:
            # This is just a place holder
            _data = torch.empty(parameter.event_shape)

        return torch.Tensor._make_subclass(cls, _data, requires_grad)

    def __init__(self, parameter=None, requires_grad=False):
        """
        The parameter class.
        :param parameter: The parameter value.
        """
        self._prior = parameter if isinstance(parameter, dist.Distribution) else None

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
        :rtype: torch.Size
        """

        return self.distr.event_shape

    def c_numel(self):
        """
        Custom 'numel' function for the number of elements in the prior's event shape.
        :rtype: int
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
        :return: Transformed distribution
        :rtype: dist.TransformedDistribution
        """

        if not self.trainable:
            raise ValueError('Is not of `Distribution` instance!')

        return dist.TransformedDistribution(self._prior, [self.bijection.inv])

    @property
    def bijection(self):
        """
        Returns a bijected function for transforms from unconstrained to constrained space.
        :rtype: callable
        """
        if not self.trainable:
            raise ValueError('Is not of `Distribution` instance!')

        return dist.biject_to(self._prior.support)

    @property
    def values(self):
        """
        Returns the actual values of the parameters.
        :rtype: float|Tensor
        """

        return self.data

    @values.setter
    def values(self, x):
        """
        Sets the values of x.
        :param x: The values
        :type x: Tensor
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
    def t_values(self):
        """
        Returns the transformed values.
        :rtype: torch.Tensor
        """

        if not self.trainable:
            raise ValueError('Cannot transform parameter not of instance `Distribution`!')

        return self.bijection.inv(self.data)

    @t_values.setter
    def t_values(self, x):
        """
        Sets transformed values.
        :param x: The values
        :type x: Tensor
        """

        self.values = self.bijection(x)

    @property
    def distr(self):
        """
        Returns the distribution.
        :rtype: torch.distributions.Distribution
        """

        if self.trainable:
            return self._prior

        raise ValueError('Does not have a distribution!')

    def sample_(self, shape=None):
        """
        Samples the variable from prior distribution in place.
        :param shape: The shape to use
        :type shape: int|tuple[int]|torch.Size
        :rtype: Parameter
        """
        if not self.trainable:
            raise ValueError('Cannot initialize parameter as it is not of instance `Distribution`!')

        self.data = self._prior.sample(size_getter(shape))

        return self

    @property
    def trainable(self):
        """
        Boolean of whether parameter is trainable.
        :rtype: bool
        """

        return isinstance(self._prior, dist.Distribution)

    def get_kde(self, weights=None, transformed=False):
        """
        Constructs KDE of the discrete representation on the transformed space.
        :param weights: The weights to use
        :type weights: torch.Tensor
        :param transformed: Whether to perform on transformed or actual values
        :type transformed: bool
        :return: KDE object
        :rtype: gaussian_kde
        """
        if transformed:
            array = self.t_values.cpu().numpy()
        else:
            array = self.values.cpu().numpy()

        weights = weights.cpu().numpy() if weights is not None else weights

        if array.ndim > 1:
            array = array[..., 0]

        return gaussian_kde(array, weights=weights)

    def get_plottable(self, num=100, **kwargs):
        """
        Gets the range and likelihood of the parameter as a `numpy.ndarray`. Used for plotting.
        :return: Range, likelihood
        :rtype: tuple[np.ndarray]
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

        xrange_ = np.linspace(low, high, num)

        return xrange_, np.exp(kde.logpdf(xrange_))

    def __reduce_ex__(self, protocol):
        return (
            _rebuild_parameter,
            (self.data, self.requires_grad, self._prior, OrderedDict())
        )
