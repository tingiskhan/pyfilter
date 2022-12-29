from pyro.distributions import Normal, MultivariateNormal
import torch

from .base import ParticleFilter
from .state import ParticleFilterPrediction, ParticleFilterState
from .utils import log_likelihood


# TODO: Add kernels
class GPF(ParticleFilter):
    """
    Implements the `Gaussian Particle Filter`_ of J.H. Kotecha and P.M. Djuric. 

    .. _`Gaussian Particle Filter`: https://ieeexplore.ieee.org/document/1232326
    """     

    def _generate_importance_density(self, state: ParticleFilterState):
        dim = -(self.ssm.hidden.n_dim + 1)
        mean = state.get_mean().unsqueeze(dim)
        chol = state.get_variance().sqrt().unsqueeze(dim) if dim == -1 else state.get_covariance().unsqueeze(dim - 1).cholesky()

        if self.ssm.hidden.n_dim == 0:
            dist = Normal(mean, chol)
        else:
            dist = MultivariateNormal(mean, chol)
        
        return dist.expand(self.particles)

    def predict(self, state: ParticleFilterState) -> ParticleFilterPrediction:        
        return ParticleFilterPrediction(state.timeseries_state, torch.ones_like(state.weights) / self.particles[-1], state.previous_indices)

    def correct(self, y: torch.Tensor, state: ParticleFilterState, prediction: ParticleFilterPrediction) -> ParticleFilterState:                
        density = self._generate_importance_density(state)
        x_vals = density.sample()

        x_copy = prediction.prev_x.copy(values=x_vals)
        x_new = self.ssm.hidden.propagate(x_copy)

        observation_density = self.ssm.build_density(x_new)
        w = observation_density.log_prob(y)

        return ParticleFilterState(x_new, w, log_likelihood(w), state.previous_indices)
