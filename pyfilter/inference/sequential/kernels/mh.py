import torch

from ....filters.base import BaseFilter

from ....constants import INFTY
from ...batch.mcmc.proposals import BaseProposal, SymmetricMH
from ...batch.mcmc.utils import run_pmmh
from ..state import SMC2State
from .base import BaseKernel


class TooManyIncreases(Exception):
    pass
    

class ParticleMetropolisHastings(BaseKernel):
    """
    Implements the Particle Metropolis Hastings kernel.
    """

    def __init__(self, num_steps=1, proposal: BaseProposal = None, distance_threshold: float = None, acceptance_threshold: float = 0.2, max_increases: int = 5, **kwargs):
        """
        Internal initializer for :class:`ParticleMetropolisHastings`.

        Args:
            num_steps (int, optional): number of successive PMMH steps to perform at each update.. Defaults to 1.
            proposal (BaseProposal, optional): method for how to generate the proposal distribution.. Defaults to None.
            distance_threshold (float, optional): when the relative distance between two consecutive PMCMC moves is lower than
            ``distance_threshold``, stop iterating. Defaults to None.
            acceptance_threshold (float, optional): if the acceptance rate falls below this number, the particles of the filter are increased.
        """

        super().__init__(**kwargs)

        self._n_steps = num_steps
        self._proposal = proposal or SymmetricMH()

        self._dist_thresh = distance_threshold
        self._is_adaptive = distance_threshold is not None

        self._acceptance_threshold = acceptance_threshold
        self._max_increases = max_increases
        self._increases = 0

    def update(self, context, filter_, state: SMC2State):
        indices = self._resampler(state.normalized_weights(), normalized=True)

        dist = self._proposal.build(context, state, filter_, state.parsed_data)

        context.resample(indices)
        state.filter_state.resample(indices)
        
        shape = torch.Size([]) if any(dist.batch_shape) else filter_.batch_shape

        # NB: The adaptive part is inspired by https://github.com/nchopin/particles
        old_params = context.stack_parameters(constrained=False)        

        with context.make_new() as sub_context:
            proposal_filter = filter_.copy()
            sub_context.set_batch_shape(context.batch_shape)
            proposal_filter.initialize_model(sub_context)

        previous_distance = 0.0
        acceptance_rate = 0.0
        for i in range(self._n_steps):
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

            acceptance_rate = (to_accept.float().mean() + i * acceptance_rate) / (i + 1)

            if acceptance_rate < self._acceptance_threshold:
                state = self._increase_states(filter_, state)
                return self.update(context, filter_, state)

            if not self._is_adaptive:
                continue

            new_params = context.stack_parameters(constrained=False)
            distance = (new_params - old_params).norm(dim=0, p=INFTY).mean()

            if (distance - previous_distance).abs() <= (self._dist_thresh * previous_distance):
                break

            previous_distance = distance

        with context.no_prior_verification():
            filter_.initialize_model(context)

        state.w.fill_(0.0)
        return state

    def _increase_states(self, filter_: BaseFilter, state: SMC2State) -> SMC2State:
        """
        Method that increases the number of state particles, called whenever the acceptance rate of
        :meth:`rejuvenate` falls below 20%.

        Args:
            state: the current state of the algorithm.

        Returns:
            The updated algorithm state.
        """

        if self._increases >= self._max_increases:
            raise TooManyIncreases(f"Configuration only allows {self._max_increases}!")

        filter_.increase_particles(2.0)
        filter_.set_batch_shape(filter_.batch_shape)

        new_filter_state = filter_.batch_filter(state.parsed_data, bar=False)

        weight = new_filter_state.loglikelihood - state.filter_state.loglikelihood
        self._increases += 1

        # TODO: This manual specification is not great...
        res = SMC2State(weight, new_filter_state)
        res.tensor_tuples = state.tensor_tuples
        res.current_iteration = state.current_iteration

        return res
