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
        self, y: torch.Tensor, x_new: TimeseriesState, x_old: TimeseriesState, kernel: Distribution
    ) -> torch.Tensor:
        likelihood = self._model.log_prob(y, x_new) + self._model.hidden.log_prob(x_new.state, x_old)
        return likelihood - kernel.log_prob(x_new.state)

    def sample_and_weight(self, y: torch.Tensor, x: TimeseriesState) -> (TimeseriesState, torch.Tensor):
        raise NotImplementedError()

    def pre_weight(self, y: torch.Tensor, x: TimeseriesState):
        """
        Pre-weights the sample, used in APF.
        :param y: The next observed value
        :param x: The previous state
        :return: The pre-weights
        """

        return self._model.log_prob(y, self._model.hidden.prop_apf(x))
