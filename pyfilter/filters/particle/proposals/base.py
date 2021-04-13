import torch
from torch.distributions import Distribution
from ....timeseries import StateSpaceModel, TimeseriesState


class Proposal(object):
    def __init__(self):
        """
        Defines a proposal object for how to draw the particles.
        """

        super().__init__()

        self._model = None  # type: StateSpaceModel

    def set_model(self, model: StateSpaceModel):
        self._model = model

        return self

    def _weight_with_kernel(
        self, y: torch.Tensor, x_new: TimeseriesState, hidden_dist: Distribution, kernel: Distribution
    ) -> torch.Tensor:
        obs_likelihood = self._model.observable.log_prob(y, x_new)
        return obs_likelihood + hidden_dist.log_prob(x_new.state) - kernel.log_prob(x_new.state)

    def sample_and_weight(self, y: torch.Tensor, x: TimeseriesState) -> (TimeseriesState, torch.Tensor):
        raise NotImplementedError()

    def pre_weight(self, y: torch.Tensor, x: TimeseriesState):
        """
        Pre-weights the sample, used in APF.
        :param y: The next observed value
        :param x: The previous state
        :return: The pre-weights
        """

        return self._model.observable.log_prob(y, self._model.hidden.prop_apf(x))
