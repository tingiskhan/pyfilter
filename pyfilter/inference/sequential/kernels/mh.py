import torch
from .base import BaseKernel
from ...batch.mcmc.proposals import BaseProposal, SymmetricMH
from ...batch.mcmc.utils import run_pmmh
from ..state import SMC2State


class ParticleMetropolisHastings(BaseKernel):
    """
    Implements the Particle Metropolis Hastings kernel.
    """

    def __init__(self, n_steps=1, proposal: BaseProposal = None, **kwargs):
        """
        Initializes the ``ParticleMetropolisHastings`` class.

        Args:
            n_steps: The number of successive PMMH steps to perform at each update.
            proposal: The method of how to generate the proposal distribution.
            kwargs: See base.
        """

        super().__init__(**kwargs)

        self._n_steps = n_steps
        self._proposal = proposal or SymmetricMH()
        self.accepted = None

    def update(self, filter_, state: SMC2State):
        indices = self._resampler(state.normalized_weights(), normalized=True)

        dist = self._proposal.build(state, filter_, state.parsed_data.values())

        filter_.resample(indices)
        state.filter_state.resample(indices)

        accepted = torch.zeros_like(state.w, dtype=torch.bool)
        shape = torch.Size([]) if any(dist.batch_shape) else filter_.n_parallel

        for _ in range(self._n_steps):
            to_accept = run_pmmh(filter_, state, self._proposal, dist, y, shape, mutate_kernel=False)
            accepted |= to_accept

        self.accepted = accepted.float().mean()
        state.w[:] = 0.0

        return self
