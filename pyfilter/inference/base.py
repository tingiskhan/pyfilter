from abc import ABC
from collections import OrderedDict

import torch
from typing import Tuple, Dict

from ..filters import BaseFilter
from .logging import DefaultLogger
from .state import AlgorithmState
from .parameter import PriorBoundParameter
from .context import ParameterContext


class BaseAlgorithm(ABC):
    """
    Abstract base class for all algorithms.
    """

    def __init__(self, filter_: BaseFilter):
        """
        Initializes the :class:`BaseFilterAlgorithm` class.

        Args:
             filter_: The filter to use for approximating the log likelihood.
        """

        super().__init__()
        self._filter = filter_
        self.context: ParameterContext = ParameterContext.get_context()

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

    def get_parameters(self, constrained=True) -> Dict[str, PriorBoundParameter]:
        """
        Gets the parameters in the current scope.

        Args:
            constrained: whether to return the constrained parameters.
        """

        return OrderedDict(self.context.get_parameters(constrained=constrained))

    def fit(self, y: torch.Tensor, logging: DefaultLogger = None) -> AlgorithmState:
        """
        Method to be overridden by derived classes. This method is intended to fit the data on the entire data set.

        Args:
            y: The data to consider, should of size ``(number of time steps, [dimension of observed space])``.
            logging: Class inherited from ``DefaultLogger`` to handle logging.
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
