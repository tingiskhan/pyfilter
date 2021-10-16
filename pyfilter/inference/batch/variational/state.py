import torch
from torch.distributions import Distribution
from .approximation import BaseApproximation
from ...state import AlgorithmState
from ....prior_module import HasPriorsModule


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

    # TODO: Not entirely correct with ``HasPriorsModule...``
    def sample_and_update_parameters(self, model: HasPriorsModule, shape: torch.Size, re_sample=False) -> Distribution:
        """
        Samples parameters from the posterior and updates the parameters of the model.

        Args:
            model: The model for which to sample parameters.
            shape: The sample shape to use.
            re_sample: Optional parameter specifying whether to re-initialize the parameters. Necessary whenever
                ``shape`` is other than current shape.

        Returns:
            Returns the parameter approximation.
        """

        param_dist = self.parameter_approximation.get_approximation()
        params = param_dist.rsample(shape)

        for p in model.parameters():
            p.detach_()

        if re_sample:
            model.sample_params(shape)

        model.update_parameters_from_tensor(params, constrained=False)

        return param_dist
