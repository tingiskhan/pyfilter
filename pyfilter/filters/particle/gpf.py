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
        mean = state.mean.unsqueeze(dim)
        var = state.var.sqrt().unsqueeze(dim) if dim == -1 else state.get_covariance().unsqueeze(dim - 1)

        if self.ssm.hidden.n_dim == 0:
            dist = Normal(mean, var)
        else:
            dist = MultivariateNormal(mean, covariance_matrix=var)
        
        return dist.expand(self.particles)

    def predict(self, state: ParticleFilterState) -> ParticleFilterPrediction:        
        return ParticleFilterPrediction(state.x, torch.ones_like(state.w) / self.particles[-1], state.prev_inds)

    def correct(self, y: torch.Tensor, state: ParticleFilterState, prediction: ParticleFilterPrediction) -> ParticleFilterState:                
        density = self._generate_importance_density(state)
        x_vals = density.sample()

        x_copy = prediction.prev_x.copy(values=x_vals)
        x_new = self.ssm.hidden.propagate(x_copy)

        observation_density = self.ssm.build_density(x_new)
        w = observation_density.log_prob(y)

        return ParticleFilterState(x_new, w, log_likelihood(w), state.prev_inds)
