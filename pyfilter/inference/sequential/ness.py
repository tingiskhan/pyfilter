from .base import SequentialParticleAlgorithm
from ...utils import get_ess
from ..kernels import OnlineKernel
from ..kernels.kde import NonShrinkingKernel, KernelDensityEstimate
from torch import isfinite
from abc import ABC
from .state import FilteringAlgorithmState


class BaseNESS(SequentialParticleAlgorithm, ABC):
    def __init__(self, filter_, particles, kde: KernelDensityEstimate = None, discrete=False):
        super().__init__(filter_, particles)

        self._kernel = OnlineKernel(kde=kde or NonShrinkingKernel(), discrete=discrete)

        if not isinstance(self._kernel, OnlineKernel):
            raise ValueError(f'Kernel must be of instance {OnlineKernel.__class__.__name__}!')

    def do_update(self, state: FilteringAlgorithmState) -> bool:
        raise NotImplementedError()

    def _update(self, y, state):
        # ===== Jitter ===== #
        if self.do_update(state):
            self._kernel.update(self.filter.ssm.theta_dists, self.filter, state.filter_state, state.w)
            state.w[:] = 0.

        # ===== Propagate filter ===== #
        fstate = self.filter.filter(y, state.filter_state)
        w = state.w + state.filter_state.get_loglikelihood()

        # ===== Log ESS ===== #
        self._logged_ess += (get_ess(state.w),)

        return FilteringAlgorithmState(w, fstate)


class NESS(BaseNESS):
    def __init__(self, filter_, particles, threshold=0.9, **kwargs):
        """
        Implements the NESS algorithm by Miguez and Crisan.
        :param kde: The kernel density estimator to use for sampling new parameters.
        """

        super().__init__(filter_, particles, **kwargs)
        self._threshold = threshold * particles

    def do_update(self, state):
        return (any(self._logged_ess) and self._logged_ess[-1] < self._threshold) or (~isfinite(state.w)).any()


class FixedWidthNESS(BaseNESS):
    def __init__(self, filter_, particles, block_len=125, **kwargs):
        """
        Implements a fixed observation width NESS which updates when the number of parsed observations is a modulo of
        `block_len`.
        :param block_len: The minimum block length to use
        """

        super().__init__(filter_, particles, **kwargs)
        self._bl = block_len

    def do_update(self, state):
        return (any(self._logged_ess) and len(self._logged_ess) % self._bl == 0) or (~isfinite(state.w)).any()