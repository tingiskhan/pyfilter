from ..filters.base import BaseFilter, enforce_tensor
from tqdm import tqdm
import warnings


class BaseAlgorithm(object):
    def __init__(self, filter_):
        """
        Implements a base class for algorithms, i.e. algorithms for inferring parameters.
        :param filter_: The filter
        :type filter_: BaseFilter
        """

        self._filter = filter_      # type: BaseFilter
        self._y = tuple()           # type: tuple[torch.Tensor]
        self._iterator = None

    @property
    def filter(self):
        """
        Returns the filter
        :rtype: BaseFilter
        """

        return self._filter

    @filter.setter
    def filter(self, x):
        """
        Sets the filter
        :param x: The new filter
        :type x: BaseFilter
        """

        if not isinstance(x, type(self.filter)):
            raise ValueError('`x` is not {:s}!'.format(type(self.filter)))

        self._filter = x

    def fit(self, y):
        """
        Fits the algorithm to data.
        :param y: The data to fit
        :type y: numpy.ndarray|pandas.DataFrame|torch.Tensor
        :return: Self
        :rtype: BaseAlgorithm
        """

        self._y = y

        raise NotImplementedError()

    def initialize(self):
        """
        Initializes the chosen algorithm.
        :return: Self
        :rtype: BaseAlgorithm
        """

        return self

    def __repr__(self):
        return str(self.__class__.__name__)


class SequentialAlgorithm(BaseAlgorithm):
    """
    Algorithm for online inference.
    """

    def _update(self, y):
        """
        The function to override by the inherited algorithm.
        :param y: The observation
        :type y: torch.Tensor
        :return: Self
        :rtype: SequentialAlgorithm
        """

        raise NotImplementedError()

    @enforce_tensor
    def update(self, y):
        """
        Performs an update using a single observation `y`.
        :param y: The observation
        :type y: numpy.ndarray|float|torch.Tensor
        :return: Self
        :rtype: SequentialAlgorithm
        """

        return self._update(y)

    def fit(self, y):
        self._iterator = tqdm(y, desc=str(self))

        for yt in self._iterator:
            self.update(yt)

        self._iterator = None

        return self


class BatchAlgorithm(BaseAlgorithm):
    """
    Algorithm for batch inference.
    """

    def _fit(self, y):
        """
        The method to override by sub-classes.
        :param y: The data in iterator format
        :type y: iterator
        :return: Self
        :rtype: BatchAlgorithm
        """

        raise NotImplementedError()

    @enforce_tensor
    def fit(self, y):
        self.initialize()._fit(y)

        return self


def experimental(func):
    def wrapper(obj, *args, **kwargs):
        warnings.warn('{:s} is an experimental algorithm, use at own risk'.format(str(obj)))

        return func(obj, *args, **kwargs)

    return wrapper