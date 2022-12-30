import torch
from typing import Union
from pyro.distributions import Normal, MultivariateNormal, Distribution

from .base import ParticleFilter
from .state import ParticleFilterPrediction, ParticleFilterState
from .utils import log_likelihood
from .proposals import GaussianProposal


# TODO: Add kernels
class GPF(ParticleFilter):
    """
    Implements the `Gaussian Particle Filter`_ of J.H. Kotecha and P.M. Djuric. 

    .. _`Gaussian Particle Filter`: https://ieeexplore.ieee.org/document/1232326
    """

    def __init__(self, model, particles: int, proposal: Union[str, GaussianProposal] = None, **kwargs):
        """
        Internal initializer for :class:`GPF`.
        """

        proposal = proposal if proposal is not None else GaussianProposal()

        assert isinstance(proposal, GaussianProposal), f"`proposal` must be of instance '{GaussianProposal}'!"
        super().__init__(model, particles, None, proposal, 0.0, **kwargs)

    def generate_predictive(self, state: ParticleFilterState) -> Distribution:
        """
        Constructs the approximation of the predictive distribution.

        Args:
            state (ParticleFilterState): current state of the particle filter.

        Returns:
            Distribution: predictive distribution distribution.
        """

        dim = -(self.ssm.hidden.n_dim + 1)
        x_new = self.ssm.hidden.propagate(state.timeseries_state)

        temp_state = ParticleFilterState(x_new, state.weights, None, None)

        mean = temp_state.get_mean().unsqueeze(dim)
        var = temp_state.get_variance().unsqueeze(dim) if dim == -1 else temp_state.get_covariance().unsqueeze(dim - 1)

        if self._model.hidden.n_dim == 0:
            dist = Normal(mean, var.sqrt())
        else:
            dist = MultivariateNormal(mean, covariance_matrix=var)
        
        return dist

    def predict(self, state: ParticleFilterState) -> ParticleFilterPrediction:        
        return ParticleFilterPrediction(state.timeseries_state, torch.ones_like(state.weights) / self.particles[-1], state.previous_indices)

    def correct(self, y: torch.Tensor, state: ParticleFilterState, prediction: ParticleFilterPrediction) -> ParticleFilterState:
        predictive_density = self.generate_predictive(state)
        x_new, weights = self.proposal.sample_and_weight(y, state, predictive_density)

        return ParticleFilterState(x_new, weights, log_likelihood(weights), state.previous_indices)
