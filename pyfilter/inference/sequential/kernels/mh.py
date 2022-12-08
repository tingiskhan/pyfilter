import torch
from .base import BaseKernel
from ...batch.mcmc.proposals import BaseProposal, SymmetricMH
from ...batch.mcmc.utils import run_pmmh
from ..state import SMC2State
from ....constants import INFTY


class ParticleMetropolisHastings(BaseKernel):
    """
    Implements the Particle Metropolis Hastings kernel.
    """

    def __init__(self, num_steps=1, proposal: BaseProposal = None, distance_threshold: float = None, **kwargs):
        """
        Initializes the :class:`ParticleMetropolisHastings` class.

        Args:
            num_steps: the number of successive PMMH steps to perform at each update.
            proposal: the method of how to generate the proposal distribution.
            distance_threshold: when the relative distance between two consecutive PMCMC moves is lower than
                ``distance_threshold``, abort. If ``None`` defaults to regular behaviour, i.e. iterating ``n_steps``.
                times.
            kwargs: see base.
        """

        super().__init__(**kwargs)

        self._n_steps = num_steps
        self._proposal = proposal or SymmetricMH()
        self.accepted = None
        self._dist_thresh = distance_threshold
        self._is_adaptive = distance_threshold is not None

    def update(self, context, filter_, state: SMC2State):
        indices = self._resampler(state.normalized_weights(), normalized=True)

        dist = self._proposal.build(context, state, filter_, state.parsed_data)

        context.resample(indices)
        state.filter_state.resample(indices)

        accepted = torch.zeros_like(state.w).bool()
        shape = torch.Size([]) if any(dist.batch_shape) else filter_.batch_shape

        # NB: The adaptive part is inspired by https://github.com/nchopin/particles
        old_params = context.stack_parameters(constrained=False)
        previous_distance = 0.0

        with context.make_new() as sub_context:
            proposal_filter = filter_.copy()
            sub_context.set_batch_shape(context.batch_shape)

            proposal_filter.initialize_model(sub_context)

        for _ in range(self._n_steps):
            to_accept = run_pmmh(
                context,
                state,
                self._proposal,
                dist,
                proposal_filter,
                sub_context,
                state.parsed_data,
                shape,
                mutate_kernel=False,
            )

            accepted |= to_accept

            if not self._is_adaptive:
                continue

            new_params: torch.Tensor = context.stack_parameters(constrained=False)
            distance = (new_params - old_params).norm(dim=0, p=INFTY).mean()

            if (distance - previous_distance).abs() <= (self._dist_thresh * previous_distance):
                break

            previous_distance = distance

        self.accepted = accepted.float().mean()
        state.w.fill_(0.0)

        return self
