from .base import BatchAlgorithm
from ..filters.base import BaseFilter, KalmanFilter, ParticleFilter
from ..resampling import systematic
from tqdm import tqdm
from .kernels import _jitter as jittering
from math import log, exp


class IteratedFilteringV2(BatchAlgorithm):
    def __init__(self, filter_, particles, iterations=30, resampler=systematic, cooling=0.1):
        """
        Implements the Iterated Filtering version 2 (IF2) algorithm by Ionides et al.
        :param filter_: The filter to use. If `filter_` is of type `ParticleFilter` and the number of particles is not
                        the same, the number of particles in `filter_` will be overridden.
        :type filter_: BaseFilter
        :param particles: The number of particles to use
        :type particles: int
        :param iterations: The number of iterations
        :type iterations: int
        :param cooling: How much of the scale to remain after all the iterations
        :type cooling: float
        """

        super().__init__(filter_)
        self._particles = particles

        if isinstance(filter_, ParticleFilter):
            self.filter._particles = particles
            self.filter._ess = 0.
        elif isinstance(filter_, KalmanFilter):
            self.filter.set_nparallel(particles)

        self._iters = iterations
        self._resampler = resampler

        if not (0 < cooling < 1):
            raise ValueError('`cooling` must be in range "(0, 1)"!')

        self._cooling = log(1 / cooling) / iterations

    def initialize(self):
        for th in self._filter.ssm.theta_dists:
            th.sample_(self._particles)

        return self

    def _fit(self, y):
        iterator = tqdm(range(self._iters))
        for m in iterator:
            # ===== Iterate over data ===== #
            iterator.set_description('{:s} - Iteration {:d}'.format(str(self), m + 1))

            self.filter.initialize()

            # TODO: Should perhaps be a dynamic setting of initial variance
            scale = 0.1 * exp(-self._cooling * m)
            for yt in y:
                # ===== Update parameters ===== #
                self.filter.ssm.p_apply(lambda x: jittering(x, scale), transformed=True)

                # ===== Perform filtering move ===== #
                self.filter.filter(yt)

                # ===== Resample ===== #
                if isinstance(self.filter, ParticleFilter):
                    weights = self.filter._w_old
                else:
                    weights = self.filter.s_ll[-1]

                inds = self._resampler(weights)
                self._filter = self.filter.resample(inds, entire_history=False)

                if isinstance(self.filter, ParticleFilter):
                    self.filter._w_old *= 0.

        return self
