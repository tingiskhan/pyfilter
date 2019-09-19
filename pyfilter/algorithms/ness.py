from .base import SequentialAlgorithm
from ..filters.base import ParticleFilter, cudawarning
from ..utils import get_ess, normalize
import torch
from ..resampling import residual
from .kernels import RegularizedKernel, ShrinkageKernel, AdaptiveShrinkageKernel
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

        self._kernel = kernel or AdaptiveShrinkageKernel()
        self._filter.set_nparallel(particles)

        # ===== Weights ===== #
        self._w_rec = torch.zeros(particles)

        # ===== Algorithm specific ===== #
        self._th = threshold
        self._resampler = resampling
        self._regularizer = RegularizedKernel().set_resampler(self._resampler)

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

    def _resample(self):
        """
        Helper method for resampling.
        :return: Self
        :rtype: NESS
        """
        if self._logged_ess[-1] < self._th / 2 * self._particles:
            self._regularizer.update(self.filter.ssm.theta_dists, self.filter, self._w_rec, self._logged_ess[-1])
        else:
            indices = self._resampler(self._w_rec)
            self.filter = self.filter.resample(indices, entire_history=False)

        self._w_rec = torch.zeros_like(self._w_rec)

        return self

    def _update(self, y):
        # ===== Resample ===== #
        if self._logged_ess[-1] < self._th * self._particles:
            self._resample()

        # TODO: Would be better to calculate mean/variance using un-resampled particles, fix
        # ===== Jitter ===== #
        self._kernel.update(self.filter.ssm.theta_dists, self.filter, self._w_rec, self._logged_ess[-1])

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
