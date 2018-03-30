from .base import Proposal


class Bootstrap(Proposal):
    """
    Implements the Bootstrap proposal. I.e. sampling from the prior distribution.
    """
    def draw(self, y, x, size=None, *args, **kwargs):
        return self._model.propagate(x)

    def weight(self, y, xn, xo, *args, **kwargs):
        return self._model.weight(y, xn)