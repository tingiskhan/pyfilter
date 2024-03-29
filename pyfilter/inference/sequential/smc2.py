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
        self._kernel = ParticleMetropolisHastings(proposal=kernel, max_increases=max_increases, **kwargs)

        if not isinstance(self._kernel, ParticleMetropolisHastings):
            raise ValueError(f"The kernel must be of instance {ParticleMetropolisHastings.__class__.__name__}!")

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
            state = self._kernel.update(self.context, self.filter, state)

        return state
