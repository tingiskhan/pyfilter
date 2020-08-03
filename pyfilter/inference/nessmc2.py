from .base import CombinedSequentialParticleAlgorithm
from .ness import NESS
from .smc2 import SMC2
import torch
from ..kde import ConstantKernel, robust_var, NormalApproximation
from .utils import stacker
from ..normalization import normalize


class NESSMC2(CombinedSequentialParticleAlgorithm):
    def __init__(self, filter_, particles, switch=500, smc2_kw=None, ness_kw=None):
        """
        Implements a hybrid of the NESS and SMC2 algorithm, as recommended in the NESS article. That is, we use the
        SMC2 algorithm for the first part of the series and then switch to NESS when it becomes too computationally
        demanding to use the SMC2.
        :param smc2_kw: Any key worded arguments to SMC2
        :param ness_kw: Any key worded arguments for NESS
        """

        super().__init__(filter_, particles, switch, first_kw=smc2_kw, second_kw=ness_kw)

    def make_first(self, filter_, particles, **kwargs):
        threshold = kwargs.pop('threshold', 0.5)
        return SMC2(filter_, particles, threshold=threshold, **kwargs)

    def make_second(self, filter_, particles, **kwargs):
        kde = kwargs.pop('kde', NormalApproximation())
        disc = kwargs.pop('discrete', True)
        return NESS(filter_, particles, kde=kde, discrete=disc, **kwargs)

    def do_on_switch(self, first: SMC2, second: NESS):
        if isinstance(self._second._kernel._kde, ConstantKernel):
            stacked = stacker(first.filter.ssm.theta_dists, lambda u: u.t_values)
            var = robust_var(stacked.concated, normalize(self._w_rec))

            bw = (1 / self._particles[0] ** 1.5 * var).sqrt()
            second._kernel._kde._bw_fac = bw

        self._first.rejuvenate()
        second._w_rec = torch.zeros_like(first._w_rec)