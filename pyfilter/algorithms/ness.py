from .base import SequentialAlgorithm
from ..filters.base import ParticleFilter, cudawarning
from ..utils import get_ess
import torch
from ..resampling import systematic, residual
from .kernels import AdaptiveShrinkageKernel, ShrinkageKernel


class NESS(SequentialAlgorithm):
    def __init__(self, filter_, particles, threshold=0.9, resampling=residual, kernel=None):
        """
        Implements the NESS alorithm by Miguez and Crisan.
        :param particles: The particles to use for approximating the density
        :type particles: int
        :param threshold: The threshold for when to resample the parameters
        :type threshold: float
        """

        cudawarning(resampling)

        super().__init__(filter_)

        self._kernel = kernel or AdaptiveShrinkageKernel()
        self._filter.set_nparallel(particles)

        # ===== Weights ===== #
        self._w_rec = torch.zeros(particles, device=self._device)

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

        for mod in [self.filter.ssm.hidden, self.filter.ssm.observable]:
            # ===== Regular parameters ===== #
            params = tuple()
            for param in mod.theta:
                if param.trainable:
                    param.sample_(self._particles)

                    shape = param.shape[1:]
                    if isinstance(self.filter, ParticleFilter):
                        shape = 1, *shape

                    var = param.view(self._particles, *shape)
                else:
                    var = param

                params += (var,)

            mod._theta_vals = params

            # ===== Distributional parameters ===== #
            pdict = dict()
            for k, v in mod.distributional_theta:
                shape = v.shape[1:]
                if isinstance(self.filter, ParticleFilter):
                    shape = 1, *shape

                pdict[k] = v.view(self._particles, *shape)

            mod.noise.__init__(**pdict)

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
