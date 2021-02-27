from abc import ABC
from torch.nn import Module
import torch
from typing import Tuple
from ..filters import BaseFilter, utils as u
from ..logging import LoggingWrapper, TqdmWrapper
from .state import AlgorithmState


class BaseAlgorithm(Module, ABC):
    def __init__(self):
        super().__init__()

    @u.enforce_tensor
    def fit(self, y: torch.Tensor, logging: LoggingWrapper = None, **kwargs) -> AlgorithmState:
        return self._fit(y, logging_wrapper=logging or TqdmWrapper(), **kwargs)

    def _fit(self, y: torch.Tensor, logging_wrapper: LoggingWrapper, **kwargs) -> AlgorithmState:
        raise NotImplementedError()

    def initialize(self, *args, **kwargs) -> AlgorithmState:
        raise NotImplementedError()

    def predict(self, steps: int, state: AlgorithmState, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def __repr__(self):
        return str(self.__class__.__name__)


class BaseFilterAlgorithm(BaseAlgorithm, ABC):
    def __init__(self, filter_: BaseFilter):
        """
        Base class for algorithms utilizing filters for inference.

        :param filter_: The filter
        """

        super().__init__()
        self._filter = filter_

    @property
    def filter(self) -> BaseFilter:
        return self._filter

    @filter.setter
    def filter(self, x: BaseFilter):
        if not isinstance(x, type(self.filter)):
            raise ValueError(f"'x' is not {self.filter}!")

        self._filter = x
