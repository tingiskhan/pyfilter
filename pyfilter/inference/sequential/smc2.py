import torch
from .base import SequentialParticleAlgorithm
from .kernels import ParticleMetropolisHastings
from ...utils import get_ess
from ...filters import ParticleFilter
from .state import SMC2State
from ..batch.mcmc.proposals import BaseProposal


class SMC2(SequentialParticleAlgorithm):
    """
    Implements the ``SMC2`` algorithm by Chopin et al.
    """

    def __init__(self, filter_, particles, threshold=0.2, kernel: BaseProposal = None, max_increases=5, **kwargs):
        """
        Initializes the ``SMC2`` class.

        Args:
            filter_: See base.
            particles: See base.
            threshold: The threshold of the relative ESS at which to perform a rejuvenation of the particles.
            kernel: Optional parameter. The kernel to use for mutating the particles.
            max_increases: Whenever the acceptance rate of the rejuvenation step falls below 20% we double the amount
                of state particles (as recommended in the original article). However, to avoid cases where there is such
                a mismatch between the observed data and the model that we just continue to get low acceptance rates, we
                allow capping the number of increases. This is especially useful if running multiple parallel instances
                to avoid one of them from hogging all resources.
            kwargs: Kwargs passed to ``pyfilter.inference.sequential.kernels.ParticleMetropolisHastings``.
        """

        super().__init__(filter_, particles)

        self.register_buffer("_threshold", torch.tensor(threshold * particles))
        self._kernel = ParticleMetropolisHastings(proposal=kernel, **kwargs)

        if not isinstance(self._kernel, ParticleMetropolisHastings):
            raise ValueError(f"The kernel must be of instance {ParticleMetropolisHastings.__class__.__name__}!")

        self.register_buffer("_max_increases", torch.tensor(max_increases, dtype=torch.int))
        self.register_buffer("_increases", torch.tensor(0, dtype=torch.int))

    def initialize(self) -> SMC2State:
        state = super(SMC2, self).initialize()

        return SMC2State(state.w, state.filter_state, state.ess)

    def update(self, y, state: SMC2State):
        state.append_data(y)

        filter_state = self.filter.filter(y, state.filter_state.latest_state)
        state.w += filter_state.get_loglikelihood()

        ess = get_ess(state.w)
        state.append_ess(ess)

        state.filter_state.append(filter_state)

        if ess < self._threshold or (~torch.isfinite(state.w)).any():
            state = self.rejuvenate(state)

        return state

    def rejuvenate(self, state: SMC2State):
        """
        Rejuvenates the particles using a PMCMC move, called whenever the relative ESS falls below ``._threshold``.

        Args:
            state: The current state of the algorithm.

        Returns:
            The updated algorithm state.
        """

        self._kernel.update(self.filter, state)

        if self._kernel.accepted < 0.2 and isinstance(self.filter, ParticleFilter):
            state = self._increase_states(state)

        return state

    def _increase_states(self, state: SMC2State) -> SMC2State:
        """
        Method that increases the number of state particles, called whenever the acceptance rate of ``.rejuvenate``
        falls below 20%.

        Args:
            state: The current state of the algorithm.

        Returns:
            The updated algorithm state.
        """

        if self._increases >= self._max_increases:
            raise Exception(f"Configuration only allows {self._max_increases}!")

        self.filter._particles[-1] *= 2
        self.filter.set_num_parallel(*self.particles)

        new_filter_state = self.filter.longfilter(state.parsed_data, bar=False)

        w = new_filter_state.loglikelihood - state.filter_state.loglikelihood
        self._increases += 1

        return SMC2State(w, new_filter_state, state.ess, parsed_data=state.parsed_data)
