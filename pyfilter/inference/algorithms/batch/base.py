from abc import ABC
import torch
from pyfilter.logging import LoggingWrapper
from ..base import BaseAlgorithm, BaseFilterAlgorithm


class BatchAlgorithm(BaseAlgorithm, ABC):
    def __init__(self, max_iter: int):
        """
        Algorithm for batch inference.
        """
        super(BatchAlgorithm, self).__init__()
        self._max_iter = int(max_iter)

    def is_converged(self, old_loss, new_loss):
        raise NotImplementedError()

    def _fit(self, y: torch.Tensor, logging_wrapper: LoggingWrapper, **kwargs):
        old_loss = torch.tensor(float('inf'))
        logging_wrapper.set_num_iter(self._max_iter)
        loss = -old_loss
        it = 0

        while not self.is_converged(old_loss, loss) and it < self._max_iter:
            old_loss = loss
            loss = self._step(y)
            logging_wrapper.do_log(it, self, y)
            it += 1

        return self

    def _step(self, y) -> float:
        raise NotImplementedError()


class BatchFilterAlgorithm(BaseFilterAlgorithm, ABC):
    def __init__(self, filter_):
        """
        Implements a class of inference algorithms using filters for inference.
        """

        super().__init__(filter_)
