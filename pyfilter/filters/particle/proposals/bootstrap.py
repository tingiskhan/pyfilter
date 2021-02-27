from .base import Proposal


class Bootstrap(Proposal):
    """
    Implements the Bootstrap proposal. I.e. sampling from the prior distribution.
    """

    def sample_and_weight(self, y, x):
        new_x = self._model.hidden.propagate(x)
        return new_x, self._model.observable.log_prob(y, new_x)
