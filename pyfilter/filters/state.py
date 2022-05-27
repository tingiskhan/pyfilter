from torch import Tensor
from abc import ABC
from stochproc.timeseries import TimeseriesState


class PredictionState(ABC):
    """
    Base class for filter predictions.
    """

    def create_state_from_prediction(self) -> "FilterState":
        """
        Method for creating an instance of ``FilterState``.
        """

        raise NotImplementedError()


class FilterState(dict, ABC):
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
        Resamples the necessary objects of ``self`` at ``indices``. Only matters when running parallel filters.

        Args:
            indices: The indices to select.
        """

        raise NotImplementedError()

    def get_loglikelihood(self) -> Tensor:
        """
        Returns the current estimate of the log likelihood, i.e. :math:`p(y_t)`.
        """

        raise NotImplementedError()

    def exchange(self, state: "FilterState", indices: Tensor):
        """
        Exchanges the necessary objects of ``self`` with ``state``. Only matters when running parallel filters.

        Args:
            state: The state to exchange with
            indices: Which indices of ``state`` to replace ``self`` with.
        """

        raise NotImplementedError()

    def get_timeseries_state(self) -> TimeseriesState:
        """
        Returns the state of the timeseries.
        """

        raise NotImplementedError()
