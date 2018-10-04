from ..filters.base import BaseFilter


class BaseAlgorithm(object):
    def __init__(self, filter_):
        """
        Implements a base class for algorithms, i.e. algorithms for inferring parameters.
        :param filter_: The filter
        :type filter_: BaseFilter
        """

        self._filter = filter_
        self._y = None          # type: numpy.ndarray

    def fit(self, y):
        """
        Fits the algorithm to data.
        :param y: The data to fit
        :type y: numpy.ndarray|pandas.DataFrame
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


class OnlineAlgorithm(BaseAlgorithm):
    """
    Algorithm for online inference.
    """

    def update(self, y):
        """
        Performs an update using a single observation `y`.
        :param y: The observation
        :type y: numpy.ndarray|float
        :return: Self
        :rtype: OnlineAlgorithm
        """

        raise NotImplementedError()

    def fit(self, y):
        for yt in y:
            self.update(yt)

        return self


class BatchAlgorithm(BaseAlgorithm):
    """
    Algorithm for batch inference.
    """
