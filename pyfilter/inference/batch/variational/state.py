import torch
from ...state import AlgorithmState
from .approximation import ParameterMeanField, StateMeanField
from torch.optim import Adadelta as Optimizer
from typing import Optional


class VariationalState(AlgorithmState):
    """
    State class for ``VariationalBayes``.
    """

    def __init__(
        self,
        converged: bool,
        loss: torch.Tensor,
        iterations: int,
        param_approx: ParameterMeanField,
        optimizer: Optimizer,
        state_approx: Optional[StateMeanField] = None,
    ):
        """
        Initializes the ``VariationalState`` class.

        Args:
             converged: Boolean indicating whether optimizer has converged.
             loss: The loss at the current iteration.
             iterations: The number of iterations that we have currently performed.
             param_approx: The parameter approximation.
             optimizer: The optimizer used in ``VariationalBayes``.
             state_approx: Optional parameter, the state approximation.
        """

        super().__init__()
        self.converged = converged
        self.loss = loss
        self.iterations = iterations
        self.optimizer = optimizer
        self.param_approx = param_approx
        self.state_approx = state_approx
