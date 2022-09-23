import threading
from collections import OrderedDict
from math import log2

import torch.quasirandom

from ..constants import EPS2


class QuasiRegistry(object):
    r"""
    Registry for quasi random engines.
    """

    _registry = threading.local()
    _registry.registry = OrderedDict([])

    @classmethod
    def add_engine(cls, dim: int, raise_if_exists: int = False) -> torch.quasirandom.SobolEngine:
        if dim in cls._registry.registry:
            if raise_if_exists:
                raise KeyError("Already have an engine with the same dimension!")

            return cls.get_engine(dim)

        engine = cls._registry.registry[dim] = torch.quasirandom.SobolEngine(dimension=dim, scramble=True)
        return engine

    @classmethod
    def get_engine(cls, dim: int) -> torch.quasirandom.SobolEngine:
        return cls._registry.registry[dim]

    @classmethod
    def sample(cls, dim: int, shape: torch.Size) -> torch.Tensor:
        numel = shape.numel()
        log2_samples = log2(numel)

        engine = cls.get_engine(dim)
        # TODO: Might have to be randomized?
        probs = engine.draw(numel) if not log2_samples.is_integer() else engine.draw_base2(int(log2(numel)))

        if shape.numel() == 1:
            probs.squeeze_(0)

        # NB: Same as in nchopin/particles to avoid "degeneracy"
        return 0.5 + (1.0 - EPS2) * (probs - 0.5)

    @classmethod
    def clear_registry(cls):
        cls._registry.registry.clear()
