import threading
from collections import OrderedDict

import torch.quasirandom as qr
import torch

from ..constants import EPS2


class _EngineContainer(object):
    r"""
    Container for QMC engines.
    """

    def __init__(self, engine: qr.SobolEngine, randomize: bool):
        """
        Initializes the :class:`_EngineContainer` class.

        Args:
            engine: quasi random engine.
            randomize: whether to randomize.
        """

        self._engine = engine
        self._randomize = randomize

    def draw(self, shape: torch.Size):
        numel = shape.numel()

        probs = self._engine.draw(numel)

        if shape.numel() == 1:
            probs.squeeze_(0)

        if self._randomize:
            # TODO: Verify below
            rands = torch.empty(probs.shape[-1], device=probs.device).uniform_()
            probs = (probs + rands).remainder(1.0)

        # NB: Same as in nchopin/particles to avoid "degeneracy"
        return 0.5 + (1.0 - EPS2) * (probs - 0.5)


class QuasiRegistry(object):
    r"""
    Registry for quasi random engines.
    """

    _registry = threading.local()
    _registry.registry = OrderedDict([])

    @classmethod
    def add_engine(cls, dim: int, randomize: bool, raise_if_exists: int = False) -> _EngineContainer:
        if dim in cls._registry.registry:
            if raise_if_exists:
                raise KeyError("Already have an engine with the same dimension!")

            return cls.get_engine(dim)

        engine = cls._registry.registry[dim] = _EngineContainer(qr.SobolEngine(dimension=dim, scramble=True), randomize)
        return engine

    @classmethod
    def get_engine(cls, dim: int) -> _EngineContainer:
        return cls._registry.registry[dim]

    @classmethod
    def sample(cls, dim: int, shape: torch.Size) -> torch.Tensor:
        engine = cls.get_engine(dim)
        return engine.draw(shape)

    @classmethod
    def clear_registry(cls):
        cls._registry.registry.clear()

    @classmethod
    def remove_engine(cls, dim: int):
        cls._registry.registry.pop(dim)
