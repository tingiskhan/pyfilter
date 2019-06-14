from torch import Tensor, distributions as dist, Size
import numpy as np
import torch
from functools import lru_cache
from scipy.stats import gaussian_kde
from collections import OrderedDict
from ..utils import MoveToHelper


# NB: This is basically the same as original, but we include the prior as well
def _rebuild_parameter(data, requires_grad, prior, backward_hooks):
    param = Parameter(data, requires_grad)
    # NB: This line exists only for backwards compatibility; the
    # general expectation is that backward_hooks is an empty
    # OrderedDict.  See Note [Don't serialize hooks]
    param._backward_hooks = backward_hooks
    param._prior = prior
    param._values = data

    return param


class Parameter(torch.nn.Parameter, MoveToHelper):
    def __new__(cls, parameter=None, requires_grad=False):
        if isinstance(parameter, torch.Tensor):
            _data = parameter
        elif not isinstance(parameter, dist.Distribution):
            _data = torch.tensor(parameter)
        else:
            # This is just a place holder
            _data = torch.Tensor([float('inf')])

        out = torch.Tensor._make_subclass(cls, _data, requires_grad)
        out._prior = parameter
        out._values = _data

        return out

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]

        result = type(self)(self.data.clone(), self.requires_grad)
        result._prior = self._prior

        memo[id(self)] = result
        return result

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

        return self._values

    @values.setter
    def values(self, x):
        """
        Sets the values of x.
        :param x: The values
        :type x: Tensor
        """
        if not isinstance(x, type(self.values)) and self.values is not None:
            raise ValueError('Is not the same type!')
        elif not self.trainable:
            self._values = x
            return

        support = self._prior.support.check(x)

        if (~support).any():
            raise ValueError('Found values outside bounds!')

        self._values = x
        self.data = self._values

    @property
    def t_values(self):
        """
        Returns the transformed values.
        :rtype: torch.Tensor
        """

        if not self.trainable:
            raise ValueError('Cannot transform parameter not of instance `Distribution`!')

        return self.bijection.inv(self.values)

    @t_values.setter
    def t_values(self, x):
        """
        Sets transformed values.
        :param x: The values
        :type x: Tensor
        """

        self.values = self.bijection(x)

    @property
    def dist(self):
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

        shape = torch.Size((shape,) if isinstance(shape, int) else shape) or Size()
        self.values = self._prior.sample(shape)

        return self

    def view_(self, shape):
        """
        In place version of `torch.view` but assumes that `shape` is to be appended.
        :param shape: The shape
        :type shape: int|tuple[int]|torch.Size
        :rtype: Parameter
        """

        self.values = self.values.view(*self.values.shape, *((shape,) if isinstance(shape, int) else shape))

        return self

    @property
    def trainable(self):
        """
        Boolean of whether parameter is trainable.
        :rtype: bool
        """

        return isinstance(self._prior, dist.Distribution)

    def get_kde(self, weights=None, transformed=True):
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

        transformed = kwargs.pop('transformed', None)
        kde = self.get_kde(transformed=True, **kwargs)

        # ===== Gets the range to plot ===== #
        vals = self.t_values.cpu().numpy()
        low = vals.min()
        high = vals.max()

        while kde.logpdf(low) > np.log(1e-3):
            low -= 1e-2

        while kde.logpdf(high) > np.log(1e-3):
            high += 1e-2

        xrange_ = torch.linspace(low, high, num)

        return self.bijection(xrange_).cpu().numpy(), np.exp(kde.logpdf(xrange_.cpu().numpy()))

    def __reduce_ex__(self, protocol):
        return (
            _rebuild_parameter,
            (self.data, self.requires_grad, self._prior, OrderedDict())
        )
