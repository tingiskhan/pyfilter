from torch import Tensor, distributions as dist, Size
import numpy as np
import torch
from functools import lru_cache
from scipy.stats import gaussian_kde


class Parameter(object):
    def __init__(self, p):
        """
        The parameter class. Serves as the base for parameters.
        :param p: The value of the parameter. Can either be numerical or distribution
        :type p: float|Tensor|dist.Distribution
        """

        self._p = p if isinstance(p, (torch.Tensor, dist.Distribution)) else torch.tensor(p)
        self._trainable = isinstance(self._p, dist.Distribution)
        self._values = None if self._trainable else self._p

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

        return dist.TransformedDistribution(self._p, [self.bijection.inv])

    @property
    def bijection(self):
        """
        Returns a bijected function for transforms from unconstrained to constrained space.
        :rtype: callable
        """
        if not self.trainable:
            raise ValueError('Is not of `Distribution` instance!')

        return dist.biject_to(self._p.support)

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

        support = self._p.support.check(x)

        if (~support).any():
            raise ValueError('Found values outside bounds!')

        self._values = x

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
            return self._p

        raise ValueError('Does not have a distribution!')

    def initialize(self, shape=None):
        """
        Initializes the variable.
        :param shape: The shape to use
        :type shape: int|tuple[int]|torch.Size
        :rtype: Parameter
        """
        if not self.trainable:
            raise ValueError('Cannot initialize parameter as it is not of instance `Distribution`!')

        self.values = self._p.sample(((shape,) if isinstance(shape, int) else shape) or Size())

        return self

    @property
    def trainable(self):
        """
        Boolean of whether parameter is trainable.
        :rtype: bool
        """

        return self._trainable

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
            array = self.t_values.numpy()
        else:
            array = self.values.numpy()

        weights = weights.numpy() if weights is not None else weights

        if array.ndim > 1:
            array = array[..., 0]

        return gaussian_kde(array, weights=weights)

    def get_plottable(self, num=100, **kwargs):
        """
        Gets the range and likelihood of the parameter as a `numpy.ndarray`. Used for plotting.
        :return: Range, likelihood
        :rtype: tuple[np.ndarray]
        """

        transformd = kwargs.pop('transformed', None)
        kde = self.get_kde(transformed=True, **kwargs)

        # ===== Gets the range to plot ===== #
        # TODO: This would optimally be done using the inverse of the CDF. However, scikit-learn does not have that and
        # scipy 1.2.0 is not currently on anaconda on it seems
        low = self.t_values.min()
        high = self.t_values.max()

        while kde.logpdf(low.numpy()) > np.log(1e-3):
            low -= 1e-2

        while kde.logpdf(high.numpy()) > np.log(1e-3):
            high += 1e-2

        xrange_ = torch.linspace(low, high, num)

        return self.bijection(xrange_).numpy(), np.exp(kde.logpdf(xrange_.numpy()))
