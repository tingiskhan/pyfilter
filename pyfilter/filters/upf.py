from ..proposals.unscented import Unscented, GlobalUnscented
from .sisr import SISR
from ..utils.utils import expanddims


class UPF(SISR):
    """
    Implements the Unscented Particle Filter of van der Merwe et al.
    """
    def __init__(self, model, particles, *args, **kwargs):
        super().__init__(model, particles, *args, proposal=Unscented(), **kwargs)

    def initialize(self):
        self._initialize_parameters()
        self._old_x = self._model.initialize(self._particles)
        self._proposal.ut.initialize(self._old_x)

        return self


class GlobalUPF(SISR):
    """
    Implements the Global UPF of Y Zhao.
    """
    def __init__(self, model, particles, *args, **kwargs):
        super().__init__(model, particles, *args, proposal=GlobalUnscented(), **kwargs)

    def initialize(self):
        self._initialize_parameters()
        self._old_x = self._model.initialize(self._particles)
        self._proposal.ut.initialize(expanddims(self._old_x.mean(axis=-1), self._old_x.ndim))

        return self