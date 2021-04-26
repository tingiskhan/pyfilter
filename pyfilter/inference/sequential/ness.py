from .base import SequentialParticleAlgorithm
from ...utils import get_ess
from .kernels import OnlineKernel, NonShrinkingKernel, KernelDensityEstimate
from torch import isfinite
from abc import ABC
from .state import FilteringAlgorithmState
from typing import Optional


class BaseNESS(SequentialParticleAlgorithm, ABC):
    def __init__(self, filter_, particles, kde: Optional[KernelDensityEstimate] = None, discrete=False):
        super().__init__(filter_, particles)

        self._kernel = OnlineKernel(kde=kde or NonShrinkingKernel(), discrete=discrete)

        if not isinstance(self._kernel, OnlineKernel):
            raise ValueError(f"Kernel must be of instance {OnlineKernel.__class__.__name__}!")

    def do_update(self, state: FilteringAlgorithmState) -> bool:
        raise NotImplementedError()

    def _update(self, y, state):
        if self.do_update(state):
            self._kernel.update(self.filter, state)

        filter_state = self.filter.filter(y, state.filter_state.latest_state)
        state.w += filter_state.get_loglikelihood()

        state.append_ess(get_ess(state.w))
        state.filter_state.append(filter_state)

        return state


class NESS(BaseNESS):
    """
    Implements the NESS algorithm by Miguez and Crisan.
    """

    def __init__(self, filter_, particles, threshold=0.95, **kwargs):
        super().__init__(filter_, particles, **kwargs)
        self._threshold = threshold * particles

    def do_update(self, state):
        return (any(state.ess) and state.ess[-1] < self._threshold) or (~isfinite(state.w)).any()


class FixedWidthNESS(BaseNESS):
    def __init__(self, filter_, particles, block_len=125, **kwargs):
        """
        Implements a fixed observation width NESS which updates when the number of parsed observations is a modulo of
        `block_len`.

        :param block_len: The minimum block length to use
        """

        super().__init__(filter_, particles, **kwargs)
        self._bl = block_len
        self._num_iters = 0

    def do_update(self, state):
        self._num_iters += 1
        return (self._num_iters % self._bl == 0) or (~isfinite(state.w)).any()
