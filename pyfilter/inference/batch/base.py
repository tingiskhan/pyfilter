from abc import ABC
import torch
from typing import Type, Dict, Any
from torch.optim import Optimizer, Adam, Adadelta
from ..base import BaseAlgorithm, BaseFilterAlgorithm
from ..state import AlgorithmState
from ..utils import Process
from ..logging import TQDMWrapper
from ...constants import EPS


class BaseBatchAlgorithm(BaseAlgorithm, ABC):
    """
    Abstract base class for batch type algorithms.
    """

    def __init__(self, max_iter: int):
        """
        Initializes the ``BaseBatchAlgorithm`` class.

        Args:
            max_iter: The maximum number of iterations of the total data to perform.
        """

        super(BaseBatchAlgorithm, self).__init__()
        self._max_iter = int(max_iter)

    def initialize(self, y: torch.Tensor, *args, **kwargs):
        raise NotImplementedError()


class BatchFilterAlgorithm(BaseFilterAlgorithm, ABC):
    """
    Abstract base class for batch type algorithms using filters in order to approximate the log likelihood.
    """

    def __init__(self, filter_, max_iter: int):
        """
        Initializes the ``BatchFilterAlgorithm`` class.

        Args:
            filter_: See base.
            max_iter: See ``BaseBatchAlgorithm``.
        """

        super(BatchFilterAlgorithm, self).__init__(filter_)
        self._max_iter = max_iter


class TQDMLossVisualiser(TQDMWrapper):
    """
    TQDM wrapper for loss based algorithms, such as ``VariationalBayes``.
    """

    def __init__(self, smoothing: float = 0.98):
        """
        Initializes the ``TQDMLossVisualiser`` class.

        Args:
            smoothing: Optional parameter. The smoothing to apply to the rolling loss:
                .. math::
                    \\tilde{\\theta_{t+1} = \\alpha \cdot \\tilde{\\theta_t} + (1 - \\alpha) \cdot \\theta_{t+1},

                where we have replaced ``smoothing`` with :math:`\alpha` for brevity, and where :math:`\\theta_t`
                denotes the loss at time :math:`t`.
        """

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
        self._tqdm_bar.set_description(self._desc_format.format(alg=self._alg, loss=self._run_avg_loss))

        self._tqdm_bar.update(1)


class OptimizationBasedAlgorithm(BaseBatchAlgorithm, ABC):
    """
    Abstract base class of batch type algorithms where an optimizer is used.
    """

    def __init__(
        self, model: Process, max_iter, optimizer: Type[Optimizer] = Adam, opt_kwargs: Dict[str, Any] = None
    ):
        """
        Initializes the ``OptimizationBasedAlgorithm`` class.

        Args:
             model: The stochastic process with unknown parameters used for modelling the data.
             max_iter: See base.
             optimizer: Optional parameter. The optimizer to use when updating the parameter estimates, should be a
                ``type`` of ``pytorch.nn.optimizers``.
             opt_kwargs: Kwargs passed to ``optimizer``.
        """

        super().__init__(max_iter)

        self._model = model

        self._opt_type = optimizer
        self.opt_kwargs = opt_kwargs or dict()

    def is_converged(self, old_loss: torch.Tensor, new_loss: torch.Tensor):
        """
        Method for checking, given new and old loss, whether the optimization has converged.

        Args:
            old_loss: The loss at the previous iteration.
            new_loss:
        """

        return ((new_loss - old_loss).abs() < EPS) & (old_loss != new_loss)

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
            logging.teardown()
