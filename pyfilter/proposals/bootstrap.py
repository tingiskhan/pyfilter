from .base import Proposal


class Bootstrap(Proposal):
    """
    Implements the Bootstrap proposal. I.e. sampling from the prior distribution.
    """

    def construct(self, y, x):
        self._kernel = lambda: self._model.hidden.propagate(x)
        return self

    def draw(self):
        return self._kernel()

    def weight(self, y, xn, xo):
        return self._model.weight(y, xn)