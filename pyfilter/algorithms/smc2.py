from .ness import NESS
from .base import experimental
from .kernels import ParticleMetropolisHastings, SymmetricMH, KernelDensitySampler
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
        self._kernel = kernel or SymmetricMH(resampling=resampling)

    def _update(self, y):
        # ===== Perform a filtering move ===== #
        self.filter.filter(y)
        self._w_rec += self.filter.s_ll[-1]

        # ===== Calculate efficient number of samples ===== #
        ess = get_ess(self._w_rec)
        self._logged_ess += (ess,)

        # ===== Rejuvenate if there are too few samples ===== #
        if ess < self._th * self._w_rec.shape[0] or (~torch.isfinite(self._w_rec)).any():
            self.rejuvenate()
            self._iterator.set_description(desc=str(self))

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
        self._kernel.update(self.filter.ssm.theta_dists, self.filter, self._w_rec)

        # ===== Update the description ===== #
        accepted = self._kernel.accepted

        self._iterator.set_description(desc='{:s} - Accepted particles is {:.1%}'.format(str(self), accepted))
        sleep(1)

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

        self.filter.set_nparallel(self._w_rec.shape[0]).initialize().longfilter(self._y, bar=False)

        # ===== Calculate new weights and replace filter ===== #
        self._w_rec = self.filter.loglikelihood - oldlogl

        return self


class SMC2FW(NESS):
    @experimental
    def __init__(self, filter_, particles, switch=200, resampling=residual, block_len=125, **kwargs):
        """
        Implements the SMC2 FW algorithm of Ajay Jasra and Yan Zhou.
        :param block_len: The block length to use
        :type block_len: int
        :param switch: When to switch to using fixed width for sampling
        :type switch: int
        :param kwargs: Kwargs to SMC2
        """
        super().__init__(filter_, particles, resampling=resampling)
        self._smc2 = SMC2(self.filter, particles, resampling=resampling, **kwargs)

        self._switch = int(switch)
        self._switched = False

        # ===== Resampling related ===== #
        self._kernel = KernelDensitySampler(resampling=resampling)
        self._bl = block_len
        self._min_th = 0.1

    def initialize(self):
        self._smc2.initialize()
        return self

    def _update(self, y):
        if len(self._y) < self._switch:
            # TODO: Better to do this on instantiation instead
            self._smc2._iterator = self._iterator
            return self._smc2.update(y)

        # ===== Perform switch ===== #
        if not self._switched:
            self._w_rec = self._smc2._w_rec
            self._switched = True
            self._logged_ess = self._smc2._logged_ess

        # ===== Check if to propagate ===== #
        if len(self._y) % self._bl == 0 or self._logged_ess[-1] < self._min_th * self._particles:
            self._kernel.update(self.filter.ssm.theta_dists, self.filter, self._w_rec)

        # ===== Perform a filtering move ===== #
        self.filter.filter(y)
        self._w_rec += self.filter.s_ll[-1]

        # ===== Calculate efficient number of samples ===== #
        self._logged_ess += (get_ess(self._w_rec),)

        return self

