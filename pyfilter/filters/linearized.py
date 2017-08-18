from .bootstrap import Bootstrap
from ..proposals.linearized import Linearized as Linz


class Linearized(Bootstrap):
    def __init__(self, model, particles, *args, **kwargs):
        super().__init__(model, particles, *args, proposal=Linz, **kwargs)
