from typing import Union

from .base import ParticleFilter
from .state import ParticleFilterPrediction, ParticleFilterCorrection
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

        # TODO: Should somehow verify that proposal uses gaussian proposal...?
        proposal = proposal if proposal is not None else GaussianProposal()        
        super().__init__(model, particles, proposal=proposal, **kwargs)

    def predict(self, state):
        return ParticleFilterPrediction(state.timeseries_state, state.weights, state.normalized_weights(), state.previous_indices)

    def correct(self, y, prediction):
        x_new, weights = self.proposal.sample_and_weight(y, prediction)

        return ParticleFilterCorrection(x_new, weights, log_likelihood(weights), prediction.indices)
