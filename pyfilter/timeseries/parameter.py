from torch import Tensor, distributions as dist, Size
from sklearn.neighbors.kde import KernelDensity


class Parameter(object):
    def __init__(self, p):
        """
        The parameter class. Serves as the base for parameters.
        :param p: The value of the parameter. Can either be numerical or distribution
        :type p: float|Tensor|dist.Distribution
        """

        self._p = p
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

    def kde(self):
        """
        Constructs KDE of the discrete representation.
        :return: KDE object
        :rtype: KernelDensity
        """

        raise NotImplementedError()



