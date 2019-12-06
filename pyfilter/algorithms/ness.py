from .base import SequentialParticleAlgorithm
from ..utils import get_ess
from .kernels import AdaptiveKernel, OnlineKernel
from ..filters import SISR


class NESS(SequentialParticleAlgorithm):
    def __init__(self, filter_, particles, kernel=None):
        """
        Implements the NESS algorithm by Miguez and Crisan.
        :param kernel: The kernel to use when propagating the parameter particles
        :type kernel: OnlineKernel
        """

        if isinstance(filter_, SISR) and filter_._th != 1.:
            raise ValueError('The filter must have `ess = 1.`!')

        super().__init__(filter_, particles)

        self._kernel = kernel or AdaptiveKernel()

        if not isinstance(self._kernel, OnlineKernel):
            raise ValueError('Kernel must be of instance {}!'.format(OnlineKernel.__class__.__name__))

    def _update(self, y):
        # ===== Jitter ===== #
        self._kernel.update(self.filter.ssm.theta_dists, self.filter, self._w_rec)

        # ===== Propagate filter ===== #
        self.filter.filter(y)
        self._w_rec += self.filter.s_ll[-1]

        # ===== Log ESS ===== #
        ess = get_ess(self._w_rec)
        self._logged_ess += (ess,)

        return self
