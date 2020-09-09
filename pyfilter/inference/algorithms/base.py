from abc import ABC
from ...filters import BaseFilter, utils as u
from ...module import Module
import torch
from typing import Tuple
from ...logging import LoggingWrapper, TqdmWrapper
from .state import AlgorithmState


class BaseAlgorithm(Module, ABC):
    def __init__(self):
        """
        Implements a base class for inference.
        """

        super().__init__()

    @u.enforce_tensor
    def fit(self, y: torch.Tensor, logging: LoggingWrapper = None, **kwargs):
        """
        Fits the algorithm to data.
        :param y: The data to fit
        :param logging: The logging wrapper
        :return: Self
        """

        return self._fit(y, logging_wrapper=logging or TqdmWrapper(), **kwargs)

    def _fit(self, y: torch.Tensor, logging_wrapper: LoggingWrapper, **kwargs):
        raise NotImplementedError()

    def initialize(self, *args, **kwargs):
        """
        Initializes the chosen algorithm.
        :return: Self
        """

        return self

    def predict(self, steps: int, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts `steps` ahead.
        :param steps: The number of steps
        :param args: Any arguments
        :param kwargs: Any keyworded arguments
        """

        raise NotImplementedError()

    def __repr__(self):
        return str(self.__class__.__name__)


class BaseFilterAlgorithm(BaseAlgorithm, ABC):
    def __init__(self, filter_: BaseFilter):
        """
        Base class for algorithms utilizing filters for inference.
        :param filter_: The filter
        :type filter_: BaseFilter
        """

        super().__init__()
        self._filter = filter_

    @property
    def filter(self) -> BaseFilter:
        """
        Returns the filter
        """

        return self._filter

    @filter.setter
    def filter(self, x: BaseFilter):
        """
        Sets the filter
        :param x: The new filter
        """

        if not isinstance(x, type(self.filter)):
            raise ValueError('`x` is not {:s}!'.format(type(self.filter)))

        self._filter = x
