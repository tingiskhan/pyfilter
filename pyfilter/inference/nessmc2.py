from .base import CombinedSequentialParticleAlgorithm
from .ness import NESS
from .smc2 import SMC2
import torch
from ..kde import ConstantKernel
from ..utils import concater


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
        kde = kwargs.pop('kde', ConstantKernel(1.))
        return NESS(filter_, particles, kde=kde, **kwargs)

    def do_on_switch(self, first: SMC2, second: NESS):
        self._first.rejuvenate()
        second._w_rec = torch.zeros_like(first._w_rec)

        variances = concater(*(p.t_values.var(dim=0) for p in first.filter.ssm.theta_dists))
        bw = (1 / self._particles[0] ** 1.5 * variances).sqrt()
        second._kernel._kde._bw_fac = bw
