from .base import SequentialParticleAlgorithm
from ...utils import get_ess
from .kernels import OnlineKernel, NonShrinkingKernel, JitterKernel
from torch import isfinite
from abc import ABC
from .state import SequentialAlgorithmState
from typing import Optional


class BaseOnlineAlgorithm(SequentialParticleAlgorithm, ABC):
    """
    Abstract base class for purely online particle algorithms.
    """

    def __init__(self, filter_, particles, kde: Optional[JitterKernel] = None, discrete=False):
        """
        Initializes the ``BaseOnlineAlgorithm`` class.

        Args:
            filter_: See base.
            particles: See base.
            kde: Optional parameter. The kernel density estimate to use when mutating particles from :math:`t`
                to :math:`t+1`. If ``None`` defaults to using
                ``pyfilter.inference.sequential.kernels.NonShrinkingKernel``.
            discrete: See ``pyfilter.inference.sequential.kernels.OnlineKernel``.
        """

        super().__init__(filter_, particles)

        self._kernel = OnlineKernel(kernel=kde or NonShrinkingKernel(), discrete=discrete)

        if not isinstance(self._kernel, OnlineKernel):
            raise ValueError(f"Kernel must be of instance {OnlineKernel.__class__.__name__}!")

    def do_update_particles(self, state: SequentialAlgorithmState) -> bool:
        """
        Method to be overridden by derived subclasses, decides when to perform an update step of particle approximation.

        Args:
            state: The current state of the algorithm

        Returns:
              A bool indicating whether to perform an update.
        """

        raise NotImplementedError()

    def update(self, y, state):
        if self.do_update_particles(state):
            self._kernel.update(self.filter, state)

        filter_state = self.filter.filter(y, state.filter_state.latest_state)
        state.w += filter_state.get_loglikelihood()

        state.append_ess(get_ess(state.w))
        state.filter_state.append(filter_state)

        return state


class NESS(BaseOnlineAlgorithm):
    """
    Implements the NESS algorithm by Miguez and Crisan, with the slight modification of allowing dynamic
    particle updates with regards to the current ESS.
    """

    def __init__(self, filter_, particles, threshold=0.9, **kwargs):
        """
        Initializes the ``NESS`` class.

        Args:
            filter_: See base.
            particles: See base.
            threshold: The relative ESS threshold at when update the particles.
        """

        super().__init__(filter_, particles, **kwargs)
        self._threshold = threshold * particles

    def do_update_particles(self, state):
        return (any(state.ess) and state.ess[-1] < self._threshold) or (~isfinite(state.w)).any()


class FixedWidthNESS(BaseOnlineAlgorithm):
    """
    Implements a fixed observation width version of ``NESS``.
    """

    def __init__(self, filter_, particles, block_len=125, **kwargs):
        """
        Initializes the ``NESS`` class.

        Args:
            filter_: See base.
            particles: See base.
            block_len: The length of the block of observations to parse before updating the particles.
        """

        super().__init__(filter_, particles, **kwargs)
        self._bl = block_len
        self._num_iters = 0

    def do_update_particles(self, state):
        self._num_iters += 1
        return (self._num_iters % self._bl == 0) or (~isfinite(state.w)).any()
