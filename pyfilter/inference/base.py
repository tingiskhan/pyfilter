from abc import ABC

import torch
from typing import Tuple

from ..filters import BaseFilter
from .logging import DefaultLogger
from .state import AlgorithmState
from .context import ParameterContext


class BaseAlgorithm(ABC):
    """
    Abstract base class for algorithms.
    """

    def __init__(self, filter_: BaseFilter):
        """
        Initializes the :class:`BaseFilterAlgorithm` class.

        Args:
             filter_: the filter to use for approximating the log likelihood.
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

    def fit(self, y: torch.Tensor, logging: DefaultLogger = None) -> AlgorithmState:
        """
        Method to be overridden by derived classes. This method is intended to fit the data on the entire data set.

        Args:
            y: the data to consider, should of size ``(timesteps, [dimension of observed space])``.
            logging: class inherited from :class:`~pyfilter.inference.logging.DefaultLogger` to handle logging.
        """

        raise NotImplementedError()

    def __repr__(self):
        return str(self.__class__.__name__)
