import torch
from torch.distributions import Distribution
from typing import Callable
from ....timeseries import Base, StateSpaceModel, NewState


class Proposal(object):
    def __init__(self, pre_weight_func: Callable[[Base, NewState], NewState] = None):
        """
        Defines a proposal object for how to draw the particles.
        """

        super().__init__()
        self._model = None  # type: StateSpaceModel
        self._pre_weight_func = pre_weight_func

    def set_model(self, model: StateSpaceModel):
        self._model = model

        return self

    def _weight_with_kernel(self, y: torch.Tensor, x_new: NewState, kernel: Distribution) -> torch.Tensor:
        y_dist = self._model.observable.build_density(x_new)
        return y_dist.log_prob(y) + x_new.dist.log_prob(x_new.values) - kernel.log_prob(x_new.values)

    def sample_and_weight(self, y: torch.Tensor, x: NewState) -> (NewState, torch.Tensor):
        raise NotImplementedError()

    def pre_weight(self, y: torch.Tensor, x: NewState):
        """
        Pre-weights the sample, used in APF.
        :param y: The next observed value
        :param x: The previous state
        :return: The pre-weights
        """
        new_state = self._pre_weight_func(self._model.hidden, x)
        y_dist = self._model.observable.build_density(new_state)

        return y_dist.log_prob(y)
