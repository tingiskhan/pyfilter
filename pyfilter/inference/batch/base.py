from abc import ABC
import torch
from typing import Type, Dict, Any, Sequence, TypeVar
from torch.optim import Optimizer, Adam
from ..base import BaseAlgorithm, BaseFilterAlgorithm
from ..state import AlgorithmState
from ..utils import Process
from ..logging import TQDMWrapper
from ...constants import EPS

TModel = TypeVar("TModel", bound=Process)


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

    def initialize(self, y: torch.Tensor) -> AlgorithmState:
        """
        Initializes the algorithm by returning an ``AlgorithmState``.

        Args:
            y: The dataset to parse, of shape ``(number of observations, [dimension of observable space])``.
        """

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
                    \\tilde{\\theta_{i+1} = \\alpha \\cdot \\tilde{\\theta_i} + (1 - \\alpha) \\cdot \\theta_{i+1},

                where we have replaced ``smoothing`` with :math:`\\alpha` for brevity, and where :math:`\\theta_i`
                denotes the loss at iteration :math:`i`.
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


# TODO: Should we really check convergence this way?
class OptimizationBasedAlgorithm(BaseBatchAlgorithm, ABC):
    """
    Abstract base class of batch type algorithms where an optimizer is used.
    """

    def __init__(self, model: TModel, max_iter, optimizer: Type[Optimizer] = Adam, opt_kwargs: Dict[str, Any] = None):
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
        self._opt_kwargs = opt_kwargs or dict()
        self.optimizer: Optimizer = None

    @property
    def model(self) -> TModel:
        """
        Returns the model used for fitting.
        """

        return self._model

    def is_converged(self, prev_loss: torch.Tensor, current_loss: torch.Tensor) -> bool:
        """
        Method for checking, given losses at iterations :math:`i-1` and :math:`i`, whether the optimization has
        converged.

        Args:
            prev_loss: The loss at the previous iteration.
            current_loss: The loss at the current iteration.

        Returns:
            ``bool`` indicating whether optimization has converged.
        """

        return ((current_loss - prev_loss).abs() < EPS) & (prev_loss != current_loss)

    def loss(self, y: torch.Tensor, state: AlgorithmState) -> torch.Tensor:
        """
        Method to be overridden by derived classes. Defines how to calculate loss given the dataset ``y``, and previous
        state ``state`` of the algorithm.

        Args:
            y: The dataset to consider when calculating the loss, of shape
                ``(number of observations, [dimension of observation space])``.
            state: The previous state of the algorithm.

        Returns:
            Returns the loss.
        """

        raise NotImplementedError()

    def construct_optimizer(self, parameters: Sequence[torch.Tensor]):
        """
        Constructs the optimizer to use.

        Args:
            parameters: The parameters to optimize.
        """

        self.optimizer = self._opt_type(parameters, **self._opt_kwargs)

    def fit(self, y: torch.Tensor, logging=None) -> AlgorithmState:
        logging = logging or TQDMLossVisualiser()
        state = self.initialize(y)

        try:
            logging.initialize(self, self._max_iter)

            while not state.converged and state.iterations < self._max_iter:
                old_loss = state.loss

                loss = self.loss(y, state)

                loss.backward()
                self.optimizer.step()

                state.loss = loss.detach()
                logging.do_log(state.iterations, state)

                state.iterations += 1
                state.converged = self.is_converged(old_loss, state.loss)

                self.optimizer.zero_grad()

            return state

        except Exception as e:
            raise e
        finally:
            logging.teardown()
