from abc import ABC
from typing import Optional

from torch import isfinite

from ..context import InferenceContext
from .base import SequentialParticleAlgorithm
from .kernels import JitterKernel, NonShrinkingKernel, OnlineKernel
from .state import SequentialAlgorithmState


class ContextNotSupported(Exception):
    pass


class BaseOnlineAlgorithm(SequentialParticleAlgorithm, ABC):
    """
    Abstract base class for purely online particle algorithms.
    """

    def __init__(self, filter_, particles, kernel: Optional[JitterKernel] = None, discrete=False, context=None):
        """
        Internal initializer for :class:`BaseOnlineAlgorithm`.

        Args:
            filter_ (BaseFilter): see :class:`SequentialParticleAlgorithm`.
            particles (int): see :class:`SequentialParticleAlgorithm`.
            kernel (Optional[JitterKernel], optional): jittering kernel to use when mutating particles from :math:`t`
            to :math:`t+1`. Defaults to None.
            discrete (bool, optional): see :class:`~pyfilter.inference.sequential.kernels.OnlineKernel`. Defaults to False.
            context (InferenceContext, optional): see :class:`SequentialParticleAlgorithm`.. Defaults to None.
        """

        super().__init__(filter_, particles, context=context)
        if not isinstance(self.context, InferenceContext):
            raise ContextNotSupported(f"Currently do not support '{self.context.__class__}'!")

        self._kernel = OnlineKernel(kernel=kernel or NonShrinkingKernel(), discrete=discrete)

        if not isinstance(self._kernel, OnlineKernel):
            raise ValueError(f"Kernel must be of instance {OnlineKernel.__class__.__name__}!")

    def do_update_particles(self, state: SequentialAlgorithmState) -> bool:
        """
        Method to be overridden by derived subclasses, decides when to perform an update step of particle approximation.

        Args:
            state (SequentialAlgorithmState): current state of the algorithm.
        """

        raise NotImplementedError()

    def _step(self, y, state):
        if self.do_update_particles(state):
            self._kernel.update(self.context, self.filter, state)

        filter_state = self.filter.filter(y, state.filter_state.latest_state, result=state.filter_state)
        state.append(filter_state)

        return state


class NESS(BaseOnlineAlgorithm):
    """
    Implements the `NESS`_ algorithm by Miguez and Crisan, with the slight modification of allowing dynamically triggered
    particle updates with regards to the current ESS.

    .. _`NESS`:  https://arxiv.org/abs/1308.1883
    """

    def __init__(self, filter_, particles, threshold=0.9, **kwargs):
        """
        Internal initializer for :class:`NESS`.

        Args:
            filter_ (BaseFilter): see :class:`SequentialParticleAlgorithm`.
            particles (_type_): see :class:`SequentialParticleAlgorithm`.
            threshold (float, optional):  relative ESS threshold at when update the particles.. Defaults to 0.9.
        """

        super().__init__(filter_, particles, **kwargs)
        self._threshold = threshold * particles

    def do_update_particles(self, state):
        ess = state.tensor_tuples["ess"]
        return (any(ess) and ess[-1] < self._threshold) or (~isfinite(state.w)).any()


class FixedWidthNESS(BaseOnlineAlgorithm):
    """
    Implements a fixed observation width version of :class:`NESS`.
    """

    def __init__(self, filter_, particles, block_len=125, **kwargs):
        """
        Internal initializer for :class:`FixedWidthNESS`.

        Args:
            filter_ (BaseFilter): see :class:`SequentialParticleAlgorithm`.
            particles (int): see :class:`SequentialParticleAlgorithm`.
            block_len (int, optional): length of the block of observations to parse before updating the particles. Defaults to 125.
        """

        super().__init__(filter_, particles, **kwargs)
        self._bl = block_len
        self._num_iterations = 0

    def do_update_particles(self, state):
        self._num_iterations += 1
        return (self._num_iterations % self._bl == 0) or (~isfinite(state.w)).any()
