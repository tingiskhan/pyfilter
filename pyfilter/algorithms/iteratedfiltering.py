from .base import BatchAlgorithm, preliminary
from ..filters.base import ParticleFilter
from ..filters import SISR
from ..resampling import residual
from tqdm import tqdm
from .kernels import _jitter as jittering
from math import log, exp
import torch


class IteratedFilteringV2(BatchAlgorithm):
    @preliminary
    def __init__(self, filter_, iterations=30, resampler=residual, cooling=0.1):
        """
        Implements the Iterated Filtering version 2 (IF2) algorithm by Ionides et al.
        :param filter_: The filter to use.
        :type filter_: ParticleFilter
        :param iterations: The number of iterations
        :type iterations: int
        :param cooling: How much of the scale to remain after all the iterations
        :type cooling: float
        """

        if not isinstance(filter_, SISR):
            raise NotImplementedError('Only works for filters of type {}!'.format(SISR.__class__.__name__))

        super().__init__(filter_)

        self._particles = self.filter._particles
        self.filter._ess = 0.

        self._iters = iterations
        self._resampler = resampler

        if not (0 < cooling < 1):
            raise ValueError('`cooling` must be in range "(0, 1)"!')

        self._cooling = log(1 / cooling) / iterations

    def initialize(self):
        self.filter.ssm.sample_params(self._particles)
        return self

    def _fit(self, y):
        iterator = tqdm(range(self._iters))
        for m in iterator:
            # ===== Iterate over data ===== #
            iterator.set_description('{:s} - Iteration {:d}'.format(str(self), m + 1))

            self.filter.reset().initialize()

            # TODO: Should perhaps be a dynamic setting of initial variance
            scale = 0.1 * exp(-self._cooling * m)
            for yt in y:
                # ===== Update parameters ===== #
                self.filter.ssm.p_apply(lambda x: jittering(x.t_values, scale), transformed=True)

                # ===== Perform filtering move ===== #
                self.filter.filter(yt)

                # ===== Resample ===== #
                inds = self._resampler(self.filter._w_old)
                self.filter.resample(inds, entire_history=False)

                self.filter._w_old = torch.zeros_like(self.filter._w_old)

        return self
