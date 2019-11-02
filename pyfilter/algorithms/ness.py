from .base import SequentialAlgorithm
from ..filters.base import ParticleFilter, cudawarning
from ..utils import get_ess, normalize
import torch
from ..resampling import residual
from .kernels import AdaptiveKernel
from ..filters import SISR


class NESS(SequentialAlgorithm):
    def __init__(self, filter_, particles, threshold=0.9, resampling=residual, kernel=None):
        """
        Implements the NESS alorithm by Miguez and Crisan.
        :param particles: The particles to use for approximating the density
        :type particles: int
        :param threshold: The threshold for when to resample the parameters
        :type threshold: float
        :param kernel: The kernel to use when propagating the parameter particles
        :type kernel: pyfilter.algorithms.kernels.BaseKernel
        """

        cudawarning(resampling)

        if isinstance(filter_, SISR) and filter_._th != 1.:
            raise ValueError('The filter must have `ess = 1.`!')

        super().__init__(filter_)

        self._kernel = kernel or AdaptiveKernel(ess=threshold, resampling=resampling)
        self._filter.set_nparallel(particles)

        # ===== Weights ===== #
        self._w_rec = torch.zeros(particles)

        # ===== ESS related ===== #
        self._logged_ess = (torch.tensor(particles, dtype=self._w_rec.dtype),)
        self._particles = particles

    def initialize(self):
        """
        Overwrites the initialization.
        :return: Self
        :rtype: NESS
        """

        self.filter.ssm.sample_params(self._particles)

        shape = (self._particles, 1) if isinstance(self.filter, ParticleFilter) else (self._particles,)
        self.filter.viewify_params(shape).initialize()

        return self

    @property
    def logged_ess(self):
        """
        Returns the logged ESS.
        :rtype: torch.Tensor
        """

        return torch.tensor(self._logged_ess)

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

    def predict(self, steps, aggregate=True, **kwargs):
        px, py = self.filter.predict(steps, aggregate=aggregate, **kwargs)

        if not aggregate:
            return px, py

        w = normalize(self._w_rec)
        wsqd = w.unsqueeze(-1)

        xm = (px * (wsqd if self.filter.ssm.hidden_ndim > 1 else w)).sum(1)
        ym = (py * (wsqd if self.filter.ssm.obs_ndim > 1 else w)).sum(1)

        return xm, ym
