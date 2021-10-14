import torch
from ...state import AlgorithmState
from .approximation import BaseApproximation


class VariationalResult(AlgorithmState):
    """
    State class for ``VariationalBayes``.
    """

    def __init__(
        self,
        converged: bool,
        loss: torch.Tensor,
        iterations: int,
        parameter_approximation: BaseApproximation,
        state_approximation: BaseApproximation = None,
    ):
        """
        Initializes the ``VariationalState`` class.

        Args:
            converged: Boolean indicating whether optimizer has converged.
            loss: The loss at the current iteration.
            iterations: The number of iterations that we have currently performed.
            parameter_approximation: The optimized parameter approximation.
            state_approximation: The optimized parameter approximation.
        """

        super().__init__()
        self.converged = converged
        self.loss = loss
        self.iterations = iterations
        self.parameter_approximation = parameter_approximation
        self.state_approximation = state_approximation
