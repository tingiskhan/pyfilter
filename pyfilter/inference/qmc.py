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
        self._rotation_vector: torch.Tensor = None

    def draw(self, shape: torch.Size) -> torch.Tensor:
        """
        Draws samples from the QMC engine.

        Args:
            shape: shape to draw.
        """

        numel = shape.numel()

        probs = self._engine.draw(numel)

        if shape.numel() == 1:
            probs.squeeze_(0)

        if self._randomize:
            # TODO: Verify below. From the Quasi MH-paper it seems as though the rotation vector should be constant
            #  across samples
            if self._rotation_vector is None:
                self._rotation_vector = torch.empty(probs.shape[-1], device=probs.device).uniform_()

            probs = (probs + self._rotation_vector).remainder(1.0)

        # NB: Same as in nchopin/particles to avoid "degeneracy"
        safe_probs = 0.5 + (1.0 - EPS2) * (probs - 0.5)

        new_shape = shape + torch.Size([self._engine.dimension] if self._engine.dimension > 1 else [])
        return safe_probs.reshape(new_shape)


class QuasiRegistry(object):
    r"""
    Registry for quasi random engines.
    """

    _registry = threading.local()
    _registry.registry = OrderedDict([])

    @classmethod
    def add_engine(cls, key: int, dim: int, randomize: bool, raise_if_exists: int = False) -> int: # noqa: F821
        """
        Adds an engine to the QMC registry.

        Args:
            key: the key to use for the engine.
            dim: dimension of to sample.
            randomize: whether to randomize the QMC points via rotation.
            raise_if_exists: raise error if trying to create an engine that already exists.

        Returns:
            Returns the key to use when sampling. The key is currently the dimension of the space.
        """

        if key in cls._registry.registry:
            if raise_if_exists:
                raise KeyError("Already have an engine with the same dimension!")

            return key

        cls._registry.registry[key] = _EngineContainer(qr.SobolEngine(dimension=dim, scramble=True), randomize)
        
        return key

    @classmethod
    def get_engine(cls, key: int) -> _EngineContainer:
        return cls._registry.registry[key]

    @classmethod
    def sample(cls, key: int, shape: torch.Size) -> torch.Tensor:
        engine = cls.get_engine(key)
        return engine.draw(shape)

    @classmethod
    def clear_registry(cls):
        cls._registry.registry.clear()

    @classmethod
    def remove_engine(cls, key: int):
        if key in cls._registry.registry:
            cls._registry.registry.pop(key)
