from .base import Proposal


class Bootstrap(Proposal):
    def draw(self, y, x, size=None, *args, **kwargs):
        return self._model.propagate(x)

    def weight(self, y, xn, xo, *args, **kwargs):
        return self._model.weight(y, xn)