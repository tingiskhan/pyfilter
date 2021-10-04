from abc import ABC
from torch.nn import Module
import torch
from typing import Tuple
from ..filters import BaseFilter
from .logging import DefaultLogger
from .state import AlgorithmState


class BaseAlgorithm(Module, ABC):
    """
    Abstract base class for all algorithms.
    """

    def __init__(self):
        super().__init__()

    def fit(self, y: torch.Tensor, logging: DefaultLogger = None, **kwargs) -> AlgorithmState:
        """
        Method to be overridden by derived classes. This method is intended to fit the data on the entire data set.

        Args:
            y: The data to consider, should of size ``(number of time steps, [dimension of observed space])``.
            logging: Class inherited from ``DefaultLogger`` to handle logging. E.g. ``VariationalBayes`` logs every
                iteration of the full dataset, whereas sequential algorithms every data point.
            kwargs: Any algorithm specific kwargs.
        """

        raise NotImplementedError()

    def initialize(self, *args, **kwargs) -> AlgorithmState:
        """
        Initializes the algorithm by returning an ``AlgorithmState``.

        Args:
            args: Any arguments required by the derived algorithm.
            kwargs: Any kwargs required by the derived algorithm.
        """

        raise NotImplementedError()

    def predict(self, steps: int, state: AlgorithmState, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given the current state of the algorithm, predict the timeseries ``steps`` steps into the future.

        Args:
            steps: The number of steps into the future to predict the timeseries.
            state: The current state of the algorithm.
            kwargs: Any algorithm specific kwargs.

        Returns:
            Returns the tuple ``(predicted x, predicted y)``, where ``x`` and ``y`` are of size
            ``(steps, [additional shapes])``.
        """

        raise NotImplementedError()

    def __repr__(self):
        return str(self.__class__.__name__)


class BaseFilterAlgorithm(BaseAlgorithm, ABC):
    """
    Abstract base class for algorithms utilizing filters for building an approximation of the log likelihood,
    :math:`\\log{\\hat{p}(y_{1:t})}`, rather than the exact likelihood.
    """

    def __init__(self, filter_: BaseFilter):
        """
        Initializes the ``BaseFilterAlgorithm`` class.

        Args:
             filter_: The filter to use for approximating the log likelihood.
        """

        super().__init__()
        self._filter = filter_

    @property
    def filter(self) -> BaseFilter:
        """
        Returns the algorithms instance of the filter.
        """

        return self._filter

    @filter.setter
    def filter(self, x: BaseFilter):
        if not isinstance(x, type(self.filter)):
            raise ValueError(f"'x' is not {self.filter}!")

        self._filter = x
