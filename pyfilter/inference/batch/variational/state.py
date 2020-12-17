from ...state import AlgorithmState
from .approximation import ParameterMeanField, StateMeanField
from torch.optim import Adadelta as Optimizer
from typing import Optional


class VariationalState(AlgorithmState):
    def __init__(
        self,
        converged: bool,
        loss: float,
        iterations: int,
        param_approx: ParameterMeanField,
        optimizer: Optimizer,
        state_approx: Optional[StateMeanField] = None,
    ):

        super().__init__()
        self.converged = converged
        self.loss = loss
        self.iterations = iterations
        self.optimizer = optimizer
        self.param_approx = param_approx
        self.state_approx = state_approx
