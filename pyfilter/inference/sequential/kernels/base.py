from abc import ABC

from ..state import SequentialAlgorithmState
from ...context import InferenceContext
from ....resampling import systematic
from ....filters import BaseFilter


class BaseKernel(ABC):
    """
    Abstract base class for kernels. Kernels are objects used by subclasses of :class:`SequentialParticleAlgorithm` for
    updating the particle approximation of the parameter posteriors.
    """

    def __init__(self, resampling=systematic):
        """
        Initializes the :class:`BaseKernel` class.

        Args:
             resampling: the resampling function to use.
        """

        self._resampler = resampling

    def update(self, context: InferenceContext, filter_: BaseFilter, state: SequentialAlgorithmState):
        """
        Method to be overridden by inherited classes. Specifies how to update the particle approximation of the
        parameter posteriors.

        Args:
            context: the parameter context.
            filter_: the filter used by the calling algorithm.
            state: the current state of the algorithm.
        """

        raise NotImplementedError()
