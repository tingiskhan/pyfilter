import torch
from typing import Union

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

    def predict(self, state: ParticleFilterState) -> ParticleFilterPrediction:        
        return ParticleFilterPrediction(state.timeseries_state, torch.ones_like(state.weights) / self.particles[-1], state.previous_indices)

    def correct(self, y: torch.Tensor, state: ParticleFilterState, prediction: ParticleFilterPrediction) -> ParticleFilterState:                
        x_new, w = self.proposal.sample_and_weight(y, state)

        return ParticleFilterState(x_new, w, log_likelihood(w), state.previous_indices)
