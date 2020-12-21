from .base import SequentialParticleAlgorithm
from .kernels import ParticleMetropolisHastings, SymmetricMH
from .kernels.mh import PropConstructor
from ...utils import get_ess, AppendableTensorList
from ...filters import ParticleFilter
from torch import isfinite
from .state import FilteringAlgorithmState
from typing import Optional
import torch


class SMC2(SequentialParticleAlgorithm):
    def __init__(
        self, filter_, particles, threshold=0.2, kernel: Optional[PropConstructor] = None, max_increases=5, **kwargs
    ):
        """
        Implements the SMC2 algorithm by Chopin et al.
        :param threshold: The threshold at which to perform MCMC rejuvenation
        :param kernel: The kernel to use when updating the parameters
        """

        super().__init__(filter_, particles)

        # ===== When and how to update ===== #
        self.register_buffer("_threshold", torch.tensor(threshold * particles))
        self._kernel = ParticleMetropolisHastings(proposal=kernel or SymmetricMH(), **kwargs)

        if not isinstance(self._kernel, ParticleMetropolisHastings):
            raise ValueError(f"The kernel must be of instance {ParticleMetropolisHastings.__class__.__name__}!")

        # ===== Some helpers to figure out whether to raise ===== #
        self.register_buffer("_max_increases", torch.tensor(max_increases, dtype=torch.int))
        self.register_buffer("_increases", torch.tensor(0, dtype=torch.int))

        # ===== Save data ===== #
        self._y = AppendableTensorList()

    def _update(self, y, state):
        # ===== Save data ===== #
        self._y.append(y)

        # ===== Perform a filtering move ===== #
        fstate = self.filter.filter(y, state.filter_state.latest_state)
        w = state.w + state.filter_state.latest_state.get_loglikelihood()

        # ===== Calculate efficient number of samples ===== #
        ess = get_ess(w)
        self._logged_ess.append(ess)

        state.filter_state.append(fstate)
        state = FilteringAlgorithmState(w, state.filter_state)

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
        self._kernel.update(self.filter, state, self._y)

        # ===== Increase states if less than 20% are accepted ===== #
        if self._kernel.accepted < 0.2 and isinstance(self.filter, ParticleFilter):
            state = self._increase_states(state)

        return state

    def _increase_states(self, state: FilteringAlgorithmState) -> FilteringAlgorithmState:
        if self._increases >= self._max_increases:
            raise Exception(f"Configuration only allows {self._max_increases}!")

        # ===== Create new filter with double the state particles ===== #
        self.filter._particles[-1] *= 2
        self.filter.set_nparallel(*self.particles)

        fstate = self.filter.longfilter(self._y, bar=False)

        # ===== Calculate new weights and replace filter ===== #
        w = fstate.loglikelihood - state.filter_state.loglikelihood
        self._increases += 1

        return FilteringAlgorithmState(w, fstate)
