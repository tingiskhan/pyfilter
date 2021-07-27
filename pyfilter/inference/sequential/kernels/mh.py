import torch
from torch.distributions import Distribution
from .base import BaseKernel
from ...utils import _construct_mvn, params_to_tensor
from ...batch.mcmc.proposals import BaseProposal
from ...batch.mcmc.utils import run_pmmh


class SymmetricMH(BaseProposal):
    def build(self, state, filter_, y):
        values = params_to_tensor(filter_.ssm, constrained=False)
        weights = state.normalized_weights()

        return _construct_mvn(values, weights, scale=1.1)  # Same scale in in particles

    def exchange(self, old: Distribution, new: Distribution, indices: torch.Tensor) -> None:
        pass


class ParticleMetropolisHastings(BaseKernel):
    def __init__(self, n_steps=1, proposal: BaseProposal = None, **kwargs):
        """
        Implements a base class for the particle Metropolis Hastings class.

        :param n_steps: The number of steps to perform
        """

        super().__init__(**kwargs)

        self._n_steps = n_steps
        self._proposal = proposal or SymmetricMH()
        self.accepted = None

    def _update(self, filter_, state, y, *args):
        prop_filt = filter_.copy()
        indices = self._resampler(state.normalized_weights(), normalized=True)

        dist = self._proposal.build(state, filter_, y)

        filter_.resample(indices)
        state.filter_state.resample(indices)

        accepted = torch.zeros_like(state.w, dtype=torch.bool)
        shape = torch.Size([]) if any(dist.batch_shape) else filter_.n_parallel

        for _ in range(self._n_steps):
            to_accept, prop_state, prop_filt = run_pmmh(filter_, state, self._proposal, dist, prop_filt, y, shape)
            accepted |= to_accept

            filter_.exchange(prop_filt, to_accept)
            state.filter_state.exchange(prop_state, to_accept)

        self.accepted = accepted.float().mean()
        state.w[:] = 0.0

        return self
