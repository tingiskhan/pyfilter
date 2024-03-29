from typing import Any, Dict, Optional

from .base import CombinedSequentialParticleAlgorithm
from .kernels import ShrinkingKernel
from .ness import NESS
from .smc2 import SMC2


class NESSMC2(CombinedSequentialParticleAlgorithm):
    """
    Implements a hybrid of the :class:`NESS`` and :class:`SMC2` algorithm, as recommended in the NESS article. That is,
    we use the :class:`SMC2` algorithm for the first part of the series and then switch to :class:`NESS` when it becomes
    too computationally demanding to use the :class`SMC2`.
    """

    def __init__(
        self,
        filter_,
        particles,
        switch=500,
        smc2_kw: Dict[str, Any] = None,
        ness_kw: Dict[str, Any] = None,
    ):
        """
        Internal initializer for :class:`NESSMC2`.

        Args:
            filter_ (BaseFilter): see :class:`CombinedSequentialParticleAlgorithm`.
            particles (int): see :class:`CombinedSequentialParticleAlgorithm`.
            switch (int, optional): see :class:`CombinedSequentialParticleAlgorithm`. Defaults to 500.
            smc2_kw (Dict[str, Any], optional): see :class:`CombinedSequentialParticleAlgorithm`. Defaults to None.
            ness_kw (Dict[str, Any], optional): see :class:`CombinedSequentialParticleAlgorithm`. Defaults to None.
        """

        super().__init__(filter_, particles, switch, first_kw=smc2_kw, second_kw=ness_kw)

    def make_first(self, filter_, context, particles, **kwargs):
        threshold = kwargs.pop("threshold", 0.5)
        return SMC2(filter_, particles, threshold=threshold, context=context, **kwargs)

    def make_second(self, filter_, context, particles, **kwargs):
        kernel = kwargs.pop("kernel", ShrinkingKernel())
        return NESS(
            filter_, particles, kernel=kernel, threshold=kwargs.pop("threshold", 0.95), context=context, **kwargs
        )

    def do_on_switch(self, first: SMC2, second: NESS, state):
        return state
