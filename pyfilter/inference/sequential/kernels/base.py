from abc import ABC
from ....resampling import systematic
from ....filters import BaseFilter
from ..state import SequentialAlgorithmState


class BaseKernel(ABC):
    """
    Abstract base class for kernels. Kernels are objects used by subclasses of ``SequentialParticleAlgorithm`` for
    updating the particle approximation of the parameter posteriors.
    """

    def __init__(self, resampling=systematic):
        """
        Initializes the ``BaseKernel`` class.

        Args:
             resampling: The resampling function to use.
        """

        self._resampler = resampling

    def update(self, filter_: BaseFilter, state: SequentialAlgorithmState):
        """
        Method to be overridden by inherited classes. Specifies how to update the particle approximation of the
        parameter posteriors.

        Args:
            filter_: The filter used by the calling algorithm.
            state: The current state of the algorithm.
            args: Any class specific arguments required by the inherited classes.
        """

        raise NotImplementedError()
