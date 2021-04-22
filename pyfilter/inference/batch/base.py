from abc import ABC
import torch
from typing import Type, Dict, Any
from torch.optim import Optimizer, Adam
from ..base import BaseAlgorithm, BaseFilterAlgorithm
from ..state import AlgorithmState
from ..utils import Process


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


class OptimizationBasedAlgorithm(BaseBatchAlgorithm, ABC):
    """
    Base class of batch type algorithms where an optimizer is used for finding parameters.
    """

    def __init__(
            self, model: Process, max_iter: int, optimizer: Type[Optimizer] = Adam, opt_kwargs: Dict[str, Any] = None
    ):
        super().__init__(max_iter)

        self._model = model

        self._opt_type = optimizer
        self.opt_kwargs = opt_kwargs or dict()

    def is_converged(self, old_loss, new_loss):
        raise NotImplementedError()

    def loss(self, y: torch.Tensor, state: AlgorithmState) -> torch.Tensor:
        """
        Method for defining the loss used in determining gradients.
        """

        raise NotImplementedError()

    def _fit(self, y: torch.Tensor, logging_wrapper, **kwargs) -> AlgorithmState:
        state = self.initialize(y, **kwargs)

        try:
            logging_wrapper.set_num_iter(self._max_iter)
            while not state.converged and state.iterations < self._max_iter:
                old_loss = state.loss

                elbo = state.loss = self.loss(y, state)

                elbo.backward()
                state.optimizer.step()

                logging_wrapper.do_log(state.iterations, self, y)

                state.iterations += 1
                state.converged = self.is_converged(old_loss, state.loss)
                state.optimizer.zero_grad()

        except Exception as e:
            logging_wrapper.close()
            raise e

        logging_wrapper.close()

        return state
