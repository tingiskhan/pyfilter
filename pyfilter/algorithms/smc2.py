from .ness import NESS
from .kernels import ParticleMetropolisHastings, SymmetricMH
from ..utils import get_ess, normalize
from ..filters.base import KalmanFilter, ParticleFilter
from time import sleep
from ..resampling import systematic, residual
import torch


class SMC2(NESS):
    def __init__(self, filter_, particles, threshold=0.2, resampling=residual, kernel=None):
        """
        Implements the SMC2 algorithm by Chopin et al.
        :param particles: The amount of particles
        :type particles: int
        :param threshold: The threshold at which to perform MCMC rejuvenation
        :type threshold: float
        :param kernel: The kernel to use
        :type kernel: ParticleMetropolisHastings
        """

        if isinstance(filter_, KalmanFilter):
            raise ValueError('`filter_` must be of instance `{:s}!'.format(ParticleFilter.__name__))

        super().__init__(filter_, particles, resampling=resampling)

        self._th = threshold
        self._kernel = kernel or SymmetricMH()
        self._kernel.set_resampler(self._resampler)

    def _update(self, y):
        # ===== Perform a filtering move ===== #
        self._y += (y,)
        self.filter.filter(y)
        self._w_rec += self.filter.s_ll[-1]

        # ===== Calculate efficient number of samples ===== #
        ess = get_ess(self._w_rec)
        self._logged_ess += (ess,)

        # ===== Rejuvenate if there are too few samples ===== #
        if ess < self._th * self._w_rec.shape[0] or (~torch.isfinite(self._w_rec)).any():
            self.rejuvenate()
            self._iterator.set_description(desc=str(self))

            self._lastrejuv = len(self._y)

        return self

    def rejuvenate(self):
        """
        Rejuvenates the particles using a PMCMC move.
        :return: Self
        :rtype: SMC2
        """

        # ===== Update the description ===== #
        self._iterator.set_description(desc='{:s} - Rejuvenating particles'.format(str(self)))
        self._kernel.set_data(self._y)
        self._kernel.update(self.filter.ssm.flat_theta_dists, self.filter, self._w_rec)

        # ===== Update the description ===== #
        accepted = self._kernel.accepted

        self._iterator.set_description(desc='{:s} - Accepted particles is {:.1%}'.format(str(self), accepted))
        sleep(1)

        # ===== Update recursive weights ===== #
        self._w_rec *= 0.

        # ===== Increase states if less than 20% are accepted ===== #
        if accepted < 0.2 and isinstance(self.filter, ParticleFilter):
            self._increase_states()

        return self

    def _increase_states(self):
        """
        Increases the number of states.
        :return: Self
        :rtype: SMC2
        """

        # ===== Create new filter with double the state particles ===== #
        oldlogl = self.filter.loglikelihood
        oldparts = self.filter.particles[-1]

        self.filter.reset()
        self.filter.particles = 2 * self.filter.particles[1]

        msg = '{:s} - Increasing number of state particles from {:d} -> {:d}'
        self._iterator.set_description(desc=msg.format(str(self), oldparts, self.filter.particles))

        self.filter.set_nparallel(self._w_rec.shape[0]).initialize().to_(self._device).longfilter(self._y, bar=False)

        # ===== Calculate new weights and replace filter ===== #
        self._w_rec = self.filter.loglikelihood - oldlogl

        return self
