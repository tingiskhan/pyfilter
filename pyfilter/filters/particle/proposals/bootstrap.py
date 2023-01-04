from .base import Proposal


class Bootstrap(Proposal):
    """
    Implements the basic bootstrap proposal, i.e. where the proposal distribution corresponds to the dynamics of the
    stochastic process. Or, more specifically we simply use :math:`p(x_t | x_{t-1})` to generate candidate solutions.
    """

    def sample_and_weight(self, y, prediction):
        new_x = self._model.hidden.propagate(prediction.get_timeseries_state())
        dist = self._model.build_density(new_x)

        return new_x, dist.log_prob(y)

    def copy(self) -> "Proposal":
        return Bootstrap()
