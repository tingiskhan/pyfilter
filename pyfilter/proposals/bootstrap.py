from .base import Proposal


class Bootstrap(Proposal):
    """
    Implements the Bootstrap proposal. I.e. sampling from the prior distribution.
    """

    def construct(self, y, x):
        self._kernel = self._model.hidden.propagate(x, as_dist=True)
        return self

    def weight(self, y, xn, xo):
        return self._model.log_prob(y, xn)