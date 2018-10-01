from torch import Tensor, distributions as dist
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
        self._values = None if self._trainable else p

    @property
    def values(self):
        """
        Returns the actual values of the parameters.
        :rtype: float|Tensor
        """

        return self._values

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



