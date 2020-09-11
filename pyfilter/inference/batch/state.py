from ..state import AlgorithmState
from .varapprox import ParameterMeanField, StateMeanField


class BatchState(AlgorithmState):
    def __init__(self, converged: bool, final_loss: float, iterations: int):
        self.converged = converged
        self.final_loss = final_loss
        self.iterations = iterations


class VariationalState(BatchState):
    def __init__(self, converged: bool, final_loss: float, iterations: int, param_approx: ParameterMeanField,
                 state_approx: StateMeanField = None):
        super().__init__(converged, final_loss, iterations)
        self.param_approx = param_approx
        self.state_approx = state_approx