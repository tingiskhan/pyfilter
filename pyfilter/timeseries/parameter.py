from torch import Tensor, distributions as dist, Size
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import numpy as np
import torch


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
    def bijection(self):
        """
        Returns a bijected function for transforms from constrained to unconstrained space.
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

    def get_kde(self, cv=4):
        """
        Constructs KDE of the discrete representation.
        :param cv: The number of cross-validations to use
        :type cv: int
        :return: KDE object
        :rtype: KernelDensity
        """
        array = self.values.numpy()

        if array.ndim < 2:
            array = array[..., None]

        grid = GridSearchCV(KernelDensity(), {'bandwidth': np.linspace(1e-6, 1, 50)}, cv=cv)
        grid = grid.fit(array)

        return KernelDensity(**grid.best_params_).fit(array)

    def get_range(self, std=4, num=100):
        """
        Gets the range of values within a given number of standard deviations.
        :param std: The number of standard deviations
        :type std: float
        :param num: The number of points in the discretized range
        :type num: int
        :rtype: np.ndarray
        """

        vals = self.t_values

        p_mean = vals.mean()
        p_std = vals.std()

        range_ = torch.linspace(p_mean - std * p_std, p_mean + std * p_std, num)

        return self.bijection(range_).numpy()

    def get_plottable(self, cv=4, std=4, num=100):
        """
        Gets the range and likelihood of the parameter as numpy. Used for plotting.
        :return: Range, likelihood
        :rtype: tuple[np.ndarray]
        """

        xrange = self.get_range(std=std, num=num).reshape(-1, 1)
        kde = self.get_kde(cv=cv)

        return xrange, np.exp(kde.score_samples(xrange))
