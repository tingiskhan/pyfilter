from .base import SequentialParticleAlgorithm
from ..kernels import ParticleMetropolisHastings, SymmetricMH
from ...utils import get_ess
from ...filters import ParticleFilter
from torch import isfinite
from .state import FilteringAlgorithmState


class SMC2(SequentialParticleAlgorithm):
    def __init__(self, filter_, particles, threshold=0.2, kernel: ParticleMetropolisHastings = None, max_increases=5):
        """
        Implements the SMC2 algorithm by Chopin et al.
        :param threshold: The threshold at which to perform MCMC rejuvenation
        :param kernel: The kernel to use when updating the parameters
        """

        super().__init__(filter_, particles)

        # ===== When and how to update ===== #
        self._threshold = threshold * particles
        self._kernel = kernel or SymmetricMH()

        if not isinstance(self._kernel, ParticleMetropolisHastings):
            raise ValueError(f'The kernel must be of instance {ParticleMetropolisHastings.__class__.__name__}!')

        # ===== Some helpers to figure out whether to raise ===== #
        self._max_increases = max_increases
        self._increases = 0

        # ===== Save data ===== #
        self._y = tuple()

    def _update(self, y, state):
        # ===== Save data ===== #
        self._y += (y,)

        # ===== Perform a filtering move ===== #
        fstate = self.filter.filter(y, state.filter_state)
        w = state.w + state.filter_state.get_loglikelihood()

        # ===== Calculate efficient number of samples ===== #
        ess = get_ess(w)
        self._logged_ess += (ess,)

        state = FilteringAlgorithmState(w, fstate)

        # ===== Rejuvenate if there are too few samples ===== #
        if ess < self._threshold or (~isfinite(state.w)).any():
            state = self.rejuvenate(state)

        return state

    def rejuvenate(self, state: FilteringAlgorithmState):
        """
        Rejuvenates the particles using a PMCMC move.
        :return: Self
        """

        # ===== Update the description ===== #
        self._kernel.set_data(self._y)
        self._kernel.update(self.filter.ssm.theta_dists, self.filter, state.filter_state, state.w)
        state.w[:] = 0.

        # ===== Increase states if less than 20% are accepted ===== #
        if self._kernel.accepted < 0.2 and isinstance(self.filter, ParticleFilter):
            state = self._increase_states()

        return state

    def _increase_states(self) -> FilteringAlgorithmState:
        """
        Increases the number of states.
        :return: Self
        """

        if self._increases >= self._max_increases:
            raise Exception(f'Configuration only allows {self._max_increases}!')

        # ===== Create new filter with double the state particles ===== #
        oldlogl = self.filter.result.loglikelihood

        self.filter.reset()
        self.filter.particles = 2 * self.filter.particles[1]
        self.filter.reset().set_nparallel(*self.particles)

        state = self.filter.longfilter(self._y, bar=False)

        # ===== Calculate new weights and replace filter ===== #
        w = self.filter.result.loglikelihood - oldlogl
        self._increases += 1

        return FilteringAlgorithmState(w, state)

    def populate_state_dict(self):
        res = super(SMC2, self).populate_state_dict()
        res.update(**{
            "_y": self._y,
            "_increases": self._increases
        })

        return res