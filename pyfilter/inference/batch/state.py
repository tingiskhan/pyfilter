from ..state import AlgorithmState
from .variational.approximation import ParameterMeanField, StateMeanField
import torch
from torch.optim import Adadelta as Optimizer
from typing import Optional
from ...filters import FilterResult


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

        self.converged = converged
        self.loss = loss
        self.iterations = iterations
        self.optimizer = optimizer
        self.param_approx = param_approx
        self.state_approx = state_approx


class PMMHState(AlgorithmState):
    def __init__(self, initial_sample: torch.Tensor, filter_result: FilterResult):
        self.samples = [initial_sample]
        self.filter_result = filter_result

    def update(self, sample: torch.Tensor):
        self.samples.append(sample)

    def as_tensor(self):
        return torch.stack(self.samples)
