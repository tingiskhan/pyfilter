from torch import Tensor
from abc import ABC
from ..timeseries import NewState
from ..state import BaseState


class BaseFilterState(BaseState, ABC):
    """
    Abstract base class for all filter states.
    """

    def get_mean(self) -> Tensor:
        """
        Returns the mean of the current filter distribution.
        """

        raise NotImplementedError()

    def get_variance(self) -> Tensor:
        """
        Returns the variance of the current filter distribution.
        """

        raise NotImplementedError()

    def resample(self, indices: Tensor):
        """
        Resamples the necessary objects of ``self``. Only matters when running parallel filters.

        Args:
            indices: The indices to select.
        """

        raise NotImplementedError()

    def get_loglikelihood(self) -> Tensor:
        """
        Returns the current estimate of the log likelihood, i.e. :math:`p(y_t)`.
        """

        raise NotImplementedError()

    def exchange(self, state: "BaseFilterState", indices: Tensor):
        """
        Exchanges the necessary objects of ``self`` with ``state``. Only matters when running parallel filters.

        Args:
            state: The state to exchange with
            indices: Which indices of ``state`` to replace ``self`` with.
        """

        raise NotImplementedError()

    def get_timeseries_state(self) -> NewState:
        """
        Returns the state of the timeseries.
        """

        raise NotImplementedError()
