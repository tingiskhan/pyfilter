from torch import Tensor
from abc import ABC
from ..timeseries import NewState
from ..state import BaseState


class BaseFilterState(BaseState, ABC):
    """
    Base state for all filter states to inherit from.
    """

    def get_mean(self) -> Tensor:
        raise NotImplementedError()

    def get_variance(self) -> Tensor:
        raise NotImplementedError()

    def resample(self, indices: Tensor):
        raise NotImplementedError()

    def get_loglikelihood(self) -> Tensor:
        raise NotImplementedError()

    def exchange(self, state, indices: Tensor):
        raise NotImplementedError()

    def get_timeseries_state(self) -> NewState:
        raise NotImplementedError()
