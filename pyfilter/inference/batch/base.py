from abc import ABC
import torch
from ...logging import LoggingWrapper
from ..base import BaseAlgorithm, BaseFilterAlgorithm
from .state import VariationalState
from ...constants import EPS


class BaseBatchAlgorithm(BaseAlgorithm, ABC):
    def __init__(self, max_iter: int):
        """
        Algorithm for batch inference.
        """
        super(BaseBatchAlgorithm, self).__init__()
        self._max_iter = int(max_iter)

    def initialize(self, y: torch.Tensor, *args, **kwargs):
        raise NotImplementedError()


class OptimizationBatchAlgorithm(BaseBatchAlgorithm, ABC):
    def is_converged(self, old_loss, new_loss):
        return ((new_loss - old_loss) ** 2) ** 0.5 < EPS

    def _fit(self, y: torch.Tensor, logging_wrapper: LoggingWrapper, **kwargs) -> VariationalState:
        state = self.initialize(y, **kwargs)

        try:
            logging_wrapper.set_num_iter(self._max_iter)
            while not state.converged and state.iterations < self._max_iter:
                old_loss = state.loss

                state = self._step(y, state)
                logging_wrapper.do_log(state.iterations, self, y)

                state.iterations += 1
                state.converged = self.is_converged(old_loss, state.loss)

        except Exception as e:
            logging_wrapper.close()
            raise e

        logging_wrapper.close()

        return state

    def _step(self, y, state: VariationalState) -> VariationalState:
        raise NotImplementedError()


class BatchFilterAlgorithm(BaseFilterAlgorithm, BaseBatchAlgorithm, ABC):
    def __init__(self, filter_, max_iter):
        """
        Implements a class of inference algorithms using filters for inference.
        """

        super(BatchFilterAlgorithm, self).__init__(filter_)
        self._max_iter = max_iter