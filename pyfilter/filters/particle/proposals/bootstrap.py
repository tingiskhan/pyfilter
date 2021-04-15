from .base import Proposal


class Bootstrap(Proposal):
    """
    Implements the Bootstrap proposal. I.e. sampling from the prior distribution.
    """

    def sample_and_weight(self, y, x):
        new_x = self._model.hidden.propagate(x)
        y_state = self._model.observable.propagate(new_x)

        return new_x, y_state.dist.log_prob(y)
