from pyro.distributions import Normal, MultivariateNormal
import torch

from .base import ParticleFilter
from .state import ParticleFilterPrediction, ParticleFilterState
from .utils import log_likelihood


class GaussianPF(ParticleFilter):
    """
    Implements the `Gaussian Particle Filter`_ of J.H. Kotecha and P.M. Djuric. 

    .. _`Gaussian Particle Filter`: https://ieeexplore.ieee.org/document/1232326
    """

    def _generate_importance_density(self, state: ParticleFilterState):
        if self.ssm.hidden.n_dim == 0:
            return Normal(state.mean, state.var.sqrt())
        
        return MultivariateNormal(state.mean, covariance_matrix=state.get_covariance())

    def predict(self, state: ParticleFilterState) -> ParticleFilterPrediction:        
        return ParticleFilterPrediction(state.x, torch.zeros_like(state.w), state.prev_inds)

    def correct(self, y: torch.Tensor, state: ParticleFilterState, prediction: ParticleFilterPrediction) -> ParticleFilterState:                
        # TODO: This is not that nice tbh...
        if (state.x.time_index == 0).all():
            state.x = self.ssm.hidden.propagate(state.x)

        # TODO: Fix batched as the axes get re-arranged
        density = self._generate_importance_density(state)
        x_vals = density.sample(self.particles[-1:])

        x_copy = prediction.prev_x.copy(values=x_vals)        
        observation_density = self.ssm.build_density(x_copy)

        x_new = self.ssm.hidden.propagate(x_copy)
        w = observation_density.log_prob(y)

        return ParticleFilterState(x_new, w, log_likelihood(w), state.prev_inds)
