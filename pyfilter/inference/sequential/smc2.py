from typing import Union

from ...filters.particle import ParticleFilter
from ..batch.mcmc.proposals import BaseProposal
from .base import SequentialParticleAlgorithm
from .kernels import ParticleMetropolisHastings
from .state import SMC2State
from .threshold import ConstantThreshold, Thresholder


class SMC2(SequentialParticleAlgorithm):
    """
    Implements the `SMC2`_ algorithm by Chopin et al.

    .. _`SMC2`: https://arxiv.org/abs/1101.1528
    """

    def __init__(
        self,
        filter_,
        particles,
        threshold: Union[float, Thresholder] = 0.2,
        kernel: BaseProposal = None,
        max_increases=5,
        context=None,
        **kwargs,
    ):
        """
        Internal initializer for :class:`SMC2`.

        Args:
            filter_ (BaseFilter): see :class:`SequentialParticleAlgorithm`.
            particles (int): see :class:`SequentialParticleAlgorithm`.
            threshold (Union[float, Thresholder], optional): threshold of the relative ESS at which to perform a rejuvenation of the particles. Defaults to 0.2.
            kernel (BaseProposal, optional): kernel to use for mutating the particle during rejuvenation. Defaults to None.
            max_increases (int, optional): max number of increases for achieving an acceptance rate of atleast 20% in the MCMC rejuvenation step. Defaults to 5.
            context (_type_, optional): _description_. Defaults to None.
        """

        super().__init__(filter_, particles, context=context)

        self._threshold = threshold if isinstance(threshold, Thresholder) else ConstantThreshold(threshold)
        self._kernel = ParticleMetropolisHastings(proposal=kernel, **kwargs)

        if not isinstance(self._kernel, ParticleMetropolisHastings):
            raise ValueError(f"The kernel must be of instance {ParticleMetropolisHastings.__class__.__name__}!")

        self._max_increases = max_increases
        self._increases = 0

    def initialize(self) -> SMC2State:
        state = super(SMC2, self).initialize()

        return SMC2State(state.w, state.filter_state)

    def _step(self, y, state: SMC2State):
        state.append_data(y)

        filter_state = self.filter.filter(y, state.filter_state.latest_state, result=state.filter_state)
        state.append(filter_state)

        any_nans = ~state.w.isfinite().all()
        ess = state.tensor_tuples["ess"]

        if ess[-1] < (self._threshold.get_threshold(len(ess) - 1) * self.particles[0]) or any_nans:
            state = self.rejuvenate(state)

        return state

    def rejuvenate(self, state: SMC2State):
        """
        Rejuvenates the particles using a PMCMC move, called whenever the relative ESS falls below :attr:`_threshold`.

        Args:
            state: the current state of the algorithm.

        Returns:
            The updated algorithm state.
        """

        self._kernel.update(self.context, self.filter, state)

        if self._kernel.accepted < 0.2 and isinstance(self.filter, ParticleFilter):
            state = self._increase_states(state)

        return state

    def _increase_states(self, state: SMC2State) -> SMC2State:
        """
        Method that increases the number of state particles, called whenever the acceptance rate of
        :meth:`rejuvenate` falls below 20%.

        Args:
            state: the current state of the algorithm.

        Returns:
            The updated algorithm state.
        """

        if self._increases >= self._max_increases:
            raise Exception(f"Configuration only allows {self._max_increases}!")

        self.filter.increase_particles(2.0)
        self.filter.set_batch_shape(self.particles)

        new_filter_state = self.filter.batch_filter(state.parsed_data, bar=False)

        w = new_filter_state.loglikelihood - state.filter_state.loglikelihood
        self._increases += 1

        res = SMC2State(w, new_filter_state)
        res.tensor_tuples = state.tensor_tuples
        res.current_iteration = state.current_iteration

        return res
