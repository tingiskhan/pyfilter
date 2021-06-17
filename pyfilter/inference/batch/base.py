from abc import ABC
import torch
from typing import Type, Dict, Any
from torch.optim import Optimizer, Adam, Adadelta
from ..base import BaseAlgorithm, BaseFilterAlgorithm
from ..state import AlgorithmState
from ..utils import Process
from ..logging import TQDMWrapper


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


class TQDMLossVisualiser(TQDMWrapper):
    def __init__(self, smoothing: float = 0.98):
        super().__init__()
        self._run_avg_loss = 0.0
        self._smoothing = smoothing

        self._desc_format = "{alg} - Loss: {loss:,.2f}"
        self._alg = None

    def initialize(self, algorithm, num_iterations):
        super(TQDMLossVisualiser, self).initialize(algorithm, num_iterations)
        self._alg = str(algorithm)

    def do_log(self, iteration, state):
        self._run_avg_loss = self._smoothing * self._run_avg_loss + (1 - self._smoothing) * state.loss
        self._tqdm.set_description(self._desc_format.format(alg=self._alg, loss=self._run_avg_loss))

        self._tqdm.update(1)


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

    def fit(self, y: torch.Tensor, logging=None, **kwargs) -> AlgorithmState:
        logging = logging or TQDMLossVisualiser()
        state = self.initialize(y, **kwargs)

        try:
            logging.initialize(self, self._max_iter)

            while not state.converged and state.iterations < self._max_iter:
                old_loss = state.loss

                elbo = self.loss(y, state)

                elbo.backward()
                state.optimizer.step()

                state.loss = elbo.detach()
                logging.do_log(state.iterations, state)

                state.iterations += 1
                state.converged = self.is_converged(old_loss, state.loss)
                state.optimizer.zero_grad()

            return state

        except Exception as e:
            raise e
        finally:
            logging.close()
