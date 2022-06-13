from torch import Tensor
from abc import ABC
from stochproc.timeseries import TimeseriesState, StateSpaceModel, result as res


class PredictionState(ABC):
    """
    Base class for filter predictions.
    """

    def create_state_from_prediction(self, model: StateSpaceModel) -> "FilterState":
        """
        Method for creating an instance of :class:`FilterState`.

        Args:
            model: the model to use for propagating.
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
            indices: the indices to select.
        """

        raise NotImplementedError()

    def get_loglikelihood(self) -> Tensor:
        """
        Returns the current estimate of the log likelihood, i.e. :math:`p(y_t)`.
        """

        raise NotImplementedError()

    def exchange(self, other: "FilterState", mask: Tensor):
        """
        Exchanges the necessary objects of ``self`` with ``state``. Only matters when running parallel filters.

        Args:
            other: the state to exchange with
            mask: mask of ``state`` to replace ``self`` with.
        """

        raise NotImplementedError()

    def get_timeseries_state(self) -> TimeseriesState:
        """
        Returns the state of the timeseries.
        """

        raise NotImplementedError()

    def predict_path(self, model: StateSpaceModel, num_steps: int) -> res.StateSpacePath:
        """
        Predicts ``num_steps`` into the future for ``model``.

        Args:
            model: the model to predict for.
            num_steps: the number of steps into the future to predict.
        """

        raise NotImplementedError()
