from ..proposals.unscented import Unscented
from .sisr import SISR


class UPF(SISR):
    def __init__(self, model, particles, *args, **kwargs):
        super().__init__(model, particles, *args, proposal=Unscented, **kwargs)

    def initialize(self):
        self._initialize_parameters()
        self._old_x = self._model.initialize(self._particles)
        self._proposal.ut.initialize(self._old_x)

        return self

