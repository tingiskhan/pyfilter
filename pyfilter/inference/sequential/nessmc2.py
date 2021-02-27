from .base import CombinedSequentialParticleAlgorithm
from .ness import NESS
from .smc2 import SMC2
from .kernels import LiuWestShrinkage
from typing import Optional, Dict, Any


class NESSMC2(CombinedSequentialParticleAlgorithm):
    def __init__(
        self,
        filter_,
        particles,
        switch=500,
        smc2_kw: Optional[Dict[str, Any]] = None,
        ness_kw: Optional[Dict[str, Any]] = None,
    ):
        """
        Implements a hybrid of the NESS and SMC2 algorithm, as recommended in the NESS article. That is, we use the
        SMC2 algorithm for the first part of the series and then switch to NESS when it becomes too computationally
        demanding to use the SMC2.

        :param smc2_kw: Any key worded arguments to SMC2
        :param ness_kw: Any key worded arguments for NESS
        """

        super().__init__(filter_, particles, switch, first_kw=smc2_kw, second_kw=ness_kw)

    def make_first(self, filter_, particles, **kwargs):
        threshold = kwargs.pop("threshold", 0.5)
        return SMC2(filter_, particles, threshold=threshold, **kwargs)

    def make_second(self, filter_, particles, **kwargs):
        kde = kwargs.pop("kde", LiuWestShrinkage())
        return NESS(filter_, particles, kde=kde, threshold=kwargs.pop("threshold", 0.95), **kwargs)

    def do_on_switch(self, first: SMC2, second: NESS, state):
        return self._first.rejuvenate(state)
