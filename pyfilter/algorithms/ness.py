from .base import SequentialAlgorithm
from ..filters.base import ParticleFilter, cudawarning
from ..utils import get_ess
import torch
from ..resampling import systematic
from .kernels import AdaptiveShrinkageKernel


class NESS(SequentialAlgorithm):
    def __init__(self, filter_, particles, threshold=0.5, resampling=systematic, kernel=AdaptiveShrinkageKernel()):
        """
        Implements the NESS alorithm by Miguez and Crisan.
        :param particles: The particles to use for approximating the density
        :type particles: int
        :param threshold: The threshold for when to resample the parameters
        :type threshold: float
        """

        cudawarning(resampling)

        super().__init__(filter_)

        self._kernel = kernel
        self._filter.set_nparallel(particles)

        # ===== Weights ===== #
        self._w_rec = torch.zeros(particles)

        # ===== Algorithm specific ===== #
        self._th = threshold
        self._resampler = resampling

        # ===== ESS related ===== #
        self._ess = particles
        self._logged_ess = tuple()

        self._particles = particles

    def initialize(self):
        """
        Overwrites the initialization.
        :return: Self
        :rtype: NESS
        """

        # ===== Initialize parameters ===== #
        for th in self._filter.ssm.flat_theta_dists:
            th.sample_(self._particles)

            # Nested filters require parameters to have an extra dimension
            if isinstance(self.filter, ParticleFilter):
                th.view_(1)

        # ===== Re-initialize distributions ===== #
        for mod in [self.filter.ssm.hidden, self.filter.ssm.observable]:
            if len(mod.distributional_theta) > 0:
                mod.noise.__init__(**mod.distributional_theta)

        self._filter.initialize()

        return self

    @property
    def logged_ess(self):
        """
        Returns the logged ESS.
        :rtype: torch.Tensor
        """

        return torch.tensor(self._logged_ess)

    def _update(self, y):
        # ===== Resample ===== #
        self._ess = get_ess(self._w_rec)

        if self._ess < self._th * self._particles:
            indices = self._resampler(self._w_rec)
            self.filter = self.filter.resample(indices, entire_history=False)

            self._w_rec *= 0.

        # ===== Log ESS ===== #
        self._logged_ess += (self._ess,)

        # ===== Jitter ===== #
        self._kernel.update(self.filter.ssm.flat_theta_dists, self.filter, self._w_rec)

        # ===== Propagate filter ===== #
        self.filter.filter(y)
        self._w_rec += self.filter.s_ll[-1]

        return self
