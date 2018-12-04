from ..filters.base import BaseFilter, enforce_tensor
from tqdm import tqdm


class BaseAlgorithm(object):
    def __init__(self, filter_):
        """
        Implements a base class for algorithms, i.e. algorithms for inferring parameters.
        :param filter_: The filter
        :type filter_: BaseFilter
        """

        self._filter = filter_      # type: BaseFilter
        self._y = tuple()           # type: tuple[torch.Tensor]

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


class SequentialAlgorithm(BaseAlgorithm):
    """
    Algorithm for online inference.
    """

    @enforce_tensor
    def update(self, y):
        """
        Performs an update using a single observation `y`.
        :param y: The observation
        :type y: numpy.ndarray|float|torch.Tensor
        :return: Self
        :rtype: SequentialAlgorithm
        """

        raise NotImplementedError()

    @enforce_tensor
    def fit(self, y):
        for yt in tqdm(y, desc=str(self.__class__.__name__)):
            self.update(yt)

        return self


class BatchAlgorithm(BaseAlgorithm):
    """
    Algorithm for batch inference.
    """
