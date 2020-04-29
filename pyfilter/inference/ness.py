from .base import SequentialParticleAlgorithm
from ..utils import get_ess
from .kernels import OnlineKernel
from ..kde import ShrinkingKernel, KernelDensityEstimate
from torch import isfinite


class NESS(SequentialParticleAlgorithm):
    def __init__(self, filter_, particles, threshold=0.9, kde=None):
        """
        Implements the NESS algorithm by Miguez and Crisan.
        :param kde: The kernel density estimator
        :type kde: KernelDensityEstimate
        """

        super().__init__(filter_, particles)

        self._kernel = OnlineKernel(kde=kde or ShrinkingKernel())
        self._threshold = threshold * particles

        if not isinstance(self._kernel, OnlineKernel):
            raise ValueError(f'Kernel must be of instance {OnlineKernel.__class__.__name__}!')

    def _update(self, y):
        # ===== Jitter ===== #
        if (any(self._logged_ess) and self._logged_ess[-1] < self._threshold) or (~isfinite(self._w_rec)).any():
            self._kernel.update(self.filter.ssm.theta_dists, self.filter, self._w_rec)

        # ===== Propagate filter ===== #
        self.filter.filter(y)
        self._w_rec += self.filter.result._loglikelihood[-1]

        # ===== Log ESS ===== #
        self._logged_ess.append(get_ess(self._w_rec))

        return self
