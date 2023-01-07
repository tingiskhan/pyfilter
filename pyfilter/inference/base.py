from abc import ABC

import torch

from ..filters import BaseFilter
from .context import InferenceContext
from .logging import DefaultLogger
from .state import AlgorithmState


class BaseAlgorithm(ABC):
    """
    Abstract base class for algorithms.
    """

    def __init__(self, filter_: BaseFilter, context: InferenceContext = None):
        """
        Internal initializer for :class:`BaseFilterAlgorithm`.

        Args:
             filter_ (BaseFilter): filter to use for approximating the log likelihood.
             context (InferenceContext): parameter context to use. If not passed, tries to get from the stack.
        """

        super().__init__()

        self._filter = filter_
        self.context: InferenceContext = context or InferenceContext.get_context()

    @property
    def filter(self) -> BaseFilter:
        """
        Returns the algorithms instance of the filter.
        """

        return self._filter

    def fit(self, y: torch.Tensor, logging: DefaultLogger = None) -> AlgorithmState:
        r"""
        Method to be overridden by derived classes. This method is intended to fit the data on the entire data set.

        Args:
            y (torch.Tensor): data to fit model with, should be of size ``{timesteps, [dimension of observed space]}``.
            logging (DefaultLogger): class inherited from :class:`~pyfilter.inference.logging.DefaultLogger` to handle logging.
        """

        raise NotImplementedError()

    def __repr__(self):
        return str(self.__class__.__name__)
