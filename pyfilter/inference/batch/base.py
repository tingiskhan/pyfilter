from abc import ABC
import torch
from ..base import BaseAlgorithm, BaseFilterAlgorithm


class BaseBatchAlgorithm(BaseAlgorithm, ABC):
    """
    Base class for batch type algorithms for parameter inference in state space models.
    """

    def __init__(self, max_iter: int):
        super(BaseBatchAlgorithm, self).__init__()
        self._max_iter = int(max_iter)

    def initialize(self, y: torch.Tensor, *args, **kwargs):
        raise NotImplementedError()


class BatchFilterAlgorithm(BaseFilterAlgorithm, ABC):
    """
    Base class for batch type algorithms using filters in order to approximate the log likelihood.
    """

    def __init__(self, filter_, max_iter):
        super(BatchFilterAlgorithm, self).__init__(filter_)
        self._max_iter = max_iter
